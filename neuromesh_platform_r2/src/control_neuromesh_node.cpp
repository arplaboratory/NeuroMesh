#include "neuromesh_platform_r2/control_neuromesh_node.h"
#include "rclcpp/rclcpp.hpp"
#include "chrono"

namespace ControlneuromeshNode {
ControlneuromeshNode :: ControlneuromeshNode(const rclcpp::NodeOptions &options): Node("control_neuromesh_node", options), 
current_side_(Eigen::Quaternionf::Identity()),
gen_(rd_())
{
	// Declare node parameters
	this->declare_parameter<std::string>("encoder_model_name", "default_encoder_model");
	this->declare_parameter<std::string>("decoder_model_name", "default_decoder_model");
	this->declare_parameter<std::string>("topic_prefix", "features_");
	this->declare_parameter<std::string>("output_topic", "gnn_output");
	this->declare_parameter<int>("decoder_cycle_length", 1000);
	this->declare_parameter<int>("encoder_cycle_length", 1000);
	this->declare_parameter<int>("encoder_await_length", 10000);
	this->declare_parameter<std::string>("id", "default_id");
	this->declare_parameter<std::string>("image_qos_profile", "default");
	this->declare_parameter<std::string>("features_qos_profile", "default");
	this->declare_parameter<std::string>("output_qos_profile", "default");
	this->declare_parameter<std::string>("agents", "");
	this->declare_parameter<bool>("to_nchw", true);
	this->declare_parameter<bool>("ints_to_floats", true);
        this->declare_parameter<float>("goal_x", 10.0);  // Default value 10.0
        this->declare_parameter<float>("goal_y", 20.0);  // Default value 20.0
        this->declare_parameter<float>("goal_z", 0.0);   // Default value 0.0

	// Get node parameters
	this->get_parameter("encoder_model_name", encoder_model_name_);
	this->get_parameter("decoder_model_name", decoder_model_name_);
	this->get_parameter("topic_prefix", topic_prefix_);
	this->get_parameter("output_topic", output_topic_);
	this->get_parameter("decoder_cycle_length", decoder_cycle_length_);
	this->get_parameter("encoder_cycle_length", encoder_cycle_length_);
	this->get_parameter("encoder_await_length", encoder_await_length_);
	this->get_parameter("id", id_);
	this->get_parameter("image_qos_profile", image_qos_profile_);
	this->get_parameter("features_qos_profile", features_qos_profile_); //for both input and output
	this->get_parameter("output_qos_profile", output_qos_profile_);
	this->get_parameter("agents", agents_);
	this->get_parameter("to_nchw", to_nchw_);

        // Get the parameter values
        hardcoded_goal_.x = this->get_parameter("goal_x").as_double();
        hardcoded_goal_.y = this->get_parameter("goal_y").as_double();
        hardcoded_goal_.z = this->get_parameter("goal_z").as_double();

	auto feature_qos = rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(features_qos_profile_));
	auto output_qos = rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(output_qos_profile_));
	feature_publisher_ = this->create_publisher<neuromesh_interfaces::msg::CommMessage>(topic_prefix_ + id_, feature_qos);
        velocity_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("robot_velocity", 10);

	//PLACEHOLDER: update available_agents
	
	all_agents = splitAgentString(agents_);
	RCLCPP_INFO(this->get_logger(), "Agents:");
	for (const auto& agent : all_agents) {
		RCLCPP_INFO(this->get_logger(), "%s", agent.c_str());
	}
	all_agents.erase(id_); //remove self from list

	available_agents = all_agents; 

	neuromesh_interfaces::msg::StateVector tmp;
	for (uint i = 0; i < 10; ++i)
	{
		tmp.state_vector.push_back(0.);
	}
	current_states_.insert(std::pair(id_, tmp));

	for (std::string id : all_agents){
		//RCLCPP_INFO(this->get_logger(), "Going through all agents to create subscriptions");
		//RCLCPP_INFO(this->get_logger(), "Id: %s", id.c_str());
		this->createSubscription(feature_subscriptions_, id, feature_qos);
	}

	for(auto it = current_states_.begin(); it != current_states_.cend(); ++it)
	{
	    std::cout << it->first << " " << it->second.state_vector.size() << "\n";
	}

	decoder_timer_ = this->create_wall_timer(std::chrono::duration<int,std::milli>(decoder_cycle_length_), std::bind(&ControlneuromeshNode::run_decoder_cycle, this));
	encoder_timer_ = this->create_wall_timer(std::chrono::duration<int,std::milli>(encoder_cycle_length_), std::bind(&ControlneuromeshNode::run_encoder_cycle, this));
	fresh_encoder_cycle = true;
}

void ControlneuromeshNode::feature_callback(const neuromesh_interfaces::msg::CommMessage::SharedPtr msg) {
	std::string uuid = msg->id;
        received_features_[uuid] = *msg;
	//RCLCPP_INFO(this->get_logger(), "Got a feature message. uuid is %s and feature length is %d", uuid, msg->tensor.data.size());
        feature_buffer_timestamp_[uuid] = rclcpp::Time(msg->timestamp).seconds();  // Store the current time as the feature timestamp
}

void ControlneuromeshNode::pos_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
	std::lock_guard<std::mutex> lock(state_mutex_);
        // Parse position and heading data from the Odometry message
        current_states_[id_].stamp = msg->header.stamp;

        current_states_[id_].state_vector.at(0) = msg->pose.pose.position.x;
        current_states_[id_].state_vector.at(1) = msg->pose.pose.position.y;
        current_states_[id_].state_vector.at(2) = msg->pose.pose.position.z;

        // Assuming heading (quaternion) is in orientation
        // You might want to convert quaternion to Euler angles (roll, pitch, yaw)
	tf2::Quaternion quat_tf;
  	geometry_msgs::msg::Quaternion quat_msg = msg->pose.pose.orientation;
    	tf2::fromMsg(quat_msg, quat_tf);
  	double r{}, p{}, y{};
  	tf2::Matrix3x3 m(quat_tf);
  	m.getRPY(r, p, y);
        current_states_[id_].state_vector.at(6) = r;
        current_states_[id_].state_vector.at(7) = p;
        current_states_[id_].state_vector.at(8) = y;

	position_updated_ = true;
    }

void ControlneuromeshNode::vel_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::unique_lock<std::mutex> lock(state_mutex_);
        
        // Wait until position has been updated or timeout occurs
        auto start_time = std::chrono::steady_clock::now();
        while (!position_updated_) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            lock.lock();
            
            auto current_time = std::chrono::steady_clock::now();
            if (current_time - start_time > max_wait_time_) {
                std::cout << "Error: Position not updated within expected time frame. Skipping encoder cycle." << std::endl;
                return; // Exit the callback without running the encoder cycle
            }
        }
	
    // Parse velocity data from the Odometry message
    current_states_[id_].state_vector.at(3) = msg->twist.twist.linear.x;
    current_states_[id_].state_vector.at(4) = msg->twist.twist.linear.y;
    current_states_[id_].state_vector.at(5) = msg->twist.twist.linear.z;

    //double timediff = std::fabs((msg->header.stamp.sec * 1.0  + msg->header.stamp.nanosec * 1e-9) - (current_states_[id_].stamp.sec * 1.0 + current_states_[id_].stamp.nanosec * 1e-9));

    //if (timediff < 0.1)
    //{
    run_encoder_cycle();

    position_updated_ = false;
	//if(!fresh_encoder_cycle){
	//	RCLCPP_DEBUG(this->get_logger(), "Skipping image, already ran encoder this cycle");
	//	return;
	//}
	//fresh_encoder_cycle = false;

	//RCLCPP_DEBUG(this->get_logger(), "Running encoder on image.");

	//startClock("encoder_inference");
	////RCLCPP_DEBUG(this->get_logger(), "Performing Inference on Encoder");
	//encoder_result = performInference(encoder_model_name_, image_tensors);
	//RCLCPP_DEBUG(this->get_logger(), "Finished performing Inference on Encoder");
	//fresh_encoder_cycle = false;
    //}
    //else
    //{
      //  RCLCPP_INFO(this->get_logger(), "Time between messages are too large! %fs", timediff);		
    //}	

}

Eigen::Vector3f ControlneuromeshNode::quaternion_to_euler(const geometry_msgs::msg::Quaternion& q) {
        // Conversion from quaternion to Euler angles
        Eigen::Quaternionf quat(q.w, q.x, q.y, q.z);
        Eigen::Vector3f euler = quat.toRotationMatrix().eulerAngles(0, 1, 2);  // Roll, Pitch, Yaw
        return euler;
    }

std::vector<float> ControlneuromeshNode::apply_current_side(const std::vector<float>& vec) const {
    Eigen::Vector3f eigen_vec(vec[0], vec[1], vec[2]);
    Eigen::Vector3f rotated = current_side_ * eigen_vec;
    return {rotated.x(), rotated.y()};  // Return only x and y
}

void ControlneuromeshNode::update_current_side() {
    if (!goal_poses_.empty() && goal_poses_.begin()->second.position.y > 0.0f) {
        current_side_ = Eigen::Quaternionf(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ()));
    } else {
        current_side_ = Eigen::Quaternionf::Identity();
    }
}

neuromesh_interfaces::msg::Tensor ControlneuromeshNode::build_obs(const std::set<std::string>& controllable_agents) {
    neuromesh_interfaces::msg::Tensor obs = neuromesh_interfaces::msg::Tensor();
    obs.shape.dims = {4, 1, 0, 3}; // [pos, vel, heading, goal] x 1 x num_agents x 3
        
    std::vector<float> data;
    for (const auto& uuid : controllable_agents) {
        if (current_states_.find(uuid) == current_states_.end()) {
            continue;
        }

        // Goal
        std::vector<float> goal = {
            hardcoded_goal_.x,
            hardcoded_goal_.y,
            hardcoded_goal_.z,
        };
        data.insert(data.end(), goal.begin(), goal.end());
        //RCLCPP_INFO(this->get_logger(), "goal.size: %d", goal.size());
        //RCLCPP_INFO(this->get_logger(), "data.size after goals: %d", data.size());
        // Position
        std::vector<float> pos = {
            current_states_[uuid].state_vector[0],
            current_states_[uuid].state_vector[1],
            current_states_[uuid].state_vector[2]
        };
        data.insert(data.end(), pos.begin(), pos.end());
        //RCLCPP_INFO(this->get_logger(), "pos.size: %d", pos.size());
        //RCLCPP_INFO(this->get_logger(), "data.size after pos: %d", data.size());
        // Velocity
        std::vector<float> vel = {
            current_states_[uuid].state_vector[3],
            current_states_[uuid].state_vector[4],
            current_states_[uuid].state_vector[5]
        };
        data.insert(data.end(), vel.begin(), vel.end());
        //RCLCPP_INFO(this->get_logger(), "vel.size: %d", vel.size());
        //RCLCPP_INFO(this->get_logger(), "data.size after vel: %d", data.size());
        // Heading
        std::vector<float> heading = {
            current_states_[uuid].state_vector[6],
            current_states_[uuid].state_vector[7],
            current_states_[uuid].state_vector[8]
        };
        data.insert(data.end(), heading.begin(), heading.end());
        //RCLCPP_INFO(this->get_logger(), "heading.size: %d", heading.size());
        //RCLCPP_INFO(this->get_logger(), "data.size after heading: %d", data.size());
        obs.shape.dims[2]++; // Increment number of agents
    }
    
    // Calculate strides
    obs.strides.resize(4);
    obs.strides[3] = sizeof(float);
    obs.strides[2] = obs.strides[3] * obs.shape.dims[3];
    obs.strides[1] = obs.strides[2] * obs.shape.dims[2];
    obs.strides[0] = obs.strides[1] * obs.shape.dims[1];

    // Convert float data to uint8
    obs.data.resize(data.size() * sizeof(float));
    std::memcpy(obs.data.data(), data.data(), obs.data.size());
    for(auto k: data) {
        //RCLCPP_INFO(this->get_logger(), "data from build_obs %f", k);
    }

    return obs;
}

neuromesh_interfaces::msg::Tensor ControlneuromeshNode::build_features_from_obs(const neuromesh_interfaces::msg::Tensor& obs) {
    neuromesh_interfaces::msg::Tensor features = neuromesh_interfaces::msg::Tensor();
    features.name = "features";
    features.shape.dims = {1, 1, obs.shape.dims[2], 8}; // 1 x 1 x num_agents x 8
    features.data_type = 9; // float32
    
    std::vector<float> data;
    std::vector<float> obs_data(obs.data.size() / sizeof(float));
    std::memcpy(obs_data.data(), obs.data.data(), obs.data.size());
    
    for (size_t i = 0; i < obs.shape.dims[2]; ++i) {
        // d_goal - pos
        for (int j = 0; j < 2; ++j) {
            data.push_back(obs_data[i*12 + j] - obs_data[i*12 + 3 + j]);
        }
        // pos
        for (int j = 0; j < 2; ++j) {
            data.push_back(obs_data[i*12 + 3 + j]);
        }
        // heading
        data.push_back(std::cos(obs_data[i*12 + 11]));
        data.push_back(std::sin(obs_data[i*12 + 11]));
        // pos + vel*heading (element-wise multiplication)
        data.push_back(obs_data[i*12 + 3] + obs_data[i*12 + 6] * std::cos(obs_data[i*12 + 11]));
        data.push_back(obs_data[i*12 + 4] + obs_data[i*12 + 7] * std::sin(obs_data[i*12 + 11]));
    }
    
    // Calculate strides
    features.strides.resize(4);
    features.strides[3] = sizeof(float);
    features.strides[2] = features.strides[3] * features.shape.dims[3];
    features.strides[1] = features.strides[2] * features.shape.dims[2];
    features.strides[0] = features.strides[1] * features.shape.dims[1];
    
    // Convert float data to uint8
    features.data.resize(data.size() * sizeof(float));
    std::memcpy(features.data.data(), data.data(), features.data.size());

    for(auto k: data) {
        //RCLCPP_INFO(this->get_logger(), "data from build_features_from_obs %f", k);
    }
    
    return features;
}

//perform inference on tensor using model called model_name
//PLACEHOLDER
std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> ControlneuromeshNode::performInference(const std::string& model_name, const std::vector<neuromesh_interfaces::msg::Tensor>& tensors)
{
	std::promise<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> prom;
	std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> r = prom.get_future();
	std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> t(1, std::make_shared<neuromesh_interfaces::msg::Tensor>());
	t[0]->result = 1; // Cannot reach engine error code
	prom.set_value(std::move(t));
	return r;
}

//Aggregates features from available agents (and itself) into a single tensor
//PLACEHOLDER
bool ControlneuromeshNode::buildDecoderTensor(
        const std::map<std::string, neuromesh_interfaces::msg::CommMessage>& agent_features, 
        neuromesh_interfaces::msg::Tensor& aggregated_tensor)
{
	return true;
}

//Adds a subscription to the feature_subscriptions_ map
void ControlneuromeshNode::createSubscription(std::map<std::string, rclcpp::Subscription<neuromesh_interfaces::msg::CommMessage>::SharedPtr>& subscription_map, std::string id, rclcpp::QoS qos)
{
	std::string topic = topic_prefix_ + id;

	//RCLCPP_INFO(this->get_logger(), "creating subscription for topic %s", topic.c_str());

	rclcpp::Subscription<neuromesh_interfaces::msg::CommMessage>::SharedPtr feature_subscription_ = 
		this->create_subscription<neuromesh_interfaces::msg::CommMessage>(
			topic,
			qos,
			std::bind(&ControlneuromeshNode::feature_callback, this, std::placeholders::_1));

	subscription_map.insert( {id, feature_subscription_} );
}

//Remove subscription form the feature_subscriptions_ map
void ControlneuromeshNode::removeSubscription(std::map<std::string, rclcpp::Subscription<neuromesh_interfaces::msg::CommMessage>::SharedPtr> subscription_map, std::string id)
{
	subscription_map.erase(id);
}

//Convert string to ROS2 QoS profile
//from https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/nvblox_ros_common/src/qos.cpp#L26
rmw_qos_profile_t ControlneuromeshNode::parseQoSString(const std::string& str)
{
  std::string profile = str;
  // Convert to upper case.
  std::transform(profile.begin(), profile.end(), profile.begin(), ::toupper);

  if (profile == "SYSTEM_DEFAULT") {
    return rmw_qos_profile_system_default;
  }
  if (profile == "DEFAULT") {
    return rmw_qos_profile_default;
  }
  if (profile == "PARAMETER_EVENTS") {
    return rmw_qos_profile_parameter_events;
  }
  if (profile == "SERVICES_DEFAULT") {
    return rmw_qos_profile_services_default;
  }
  if (profile == "PARAMETERS") {
    return rmw_qos_profile_parameters;
  }
  if (profile == "SENSOR_DATA") {
    return rmw_qos_profile_sensor_data;
  }
  RCLCPP_WARN_STREAM(
    rclcpp::get_logger("parseQosString"),
    "Unknown QoS profile: " << profile << ". Returning profile: DEFAULT");
  return rmw_qos_profile_default;
}

//Split agent string parameter into vector of agent ids
std::set<std::string> ControlneuromeshNode::splitAgentString(std::string str)
{
	std::set<std::string> agents;
	const std::string delimiter = ",";

	size_t pos = 0;
	std::string token;
	while ((pos = str.find(delimiter)) != std::string::npos) {
		token = str.substr(0, pos);
		agents.insert(token);
		str.erase(0, pos + delimiter.length());
	}
	agents.insert(str);
	return agents;
}

void ControlneuromeshNode::run_encoder_cycle() {
    //RCLCPP_INFO(this->get_logger(), "Entering encoder cycle");
    if (fresh_encoder_cycle && !current_states_.empty()) {
        fresh_encoder_cycle = false;
        startClock("encoder_inference");

        // Step 1: Get pos, vel, goal -> build observations
        std::set<std::string> controllable_agents;
        controllable_agents.insert(current_states_.begin()->first);
        obs = build_obs(controllable_agents);
        neuromesh_interfaces::msg::Tensor features = build_features_from_obs(obs);
        //RCLCPP_INFO(this->get_logger(), "My input to the encoder looks like.... size=%d",features.data.size());
        // Step 2: Perform encoder inference
        encoder_result = performInference(encoder_model_name_, {features});
    }

    if (!fresh_encoder_cycle && encoder_result.valid()) {
        auto encoder_status = encoder_result.wait_for(std::chrono::milliseconds(0));
        if (encoder_status == std::future_status::ready) {
            stopClock("encoder_inference");
            auto encoded_features = encoder_result.get()[0];

            neuromesh_interfaces::msg::CommMessage feature_msg;
            feature_msg.id = current_states_.begin()->first;  // Assuming single agent for simplicity
            feature_msg.tensor = *encoded_features;
            //RCLCPP_INFO(this->get_logger(), "MY encoded features look like.... size = %d", encoded_features->data.size()); 
            for (auto xx : encoded_features->shape.dims){
	        //RCLCPP_INFO(this->get_logger(), "Dim = %d", xx);
	    }
	    feature_msg.timestamp = this->now();
            feature_msg.pos.x = obs.data[2];  // First agent's x position
            feature_msg.pos.y = obs.data[3];  // First agent's y position
            // Step 3: Publish features
            feature_publisher_->publish(feature_msg);

            received_features_[feature_msg.id] = feature_msg;  // Include own features
			feature_buffer_timestamp_[this->id_] = rclcpp::Time(feature_msg.timestamp).seconds();

            fresh_encoder_cycle = true;
            RCLCPP_DEBUG(this->get_logger(), "Encoder cycle completed in %lims", times["encoder_inference"].first);
        } else if (checkClock("encoder_inference") > encoder_await_length_) {
            RCLCPP_WARN(this->get_logger(), "Encoder inference timed out");
            fresh_encoder_cycle = true;
        }
	//RCLCPP_INFO(this->get_logger(), "Exiting encoder cycle");
    }
}

// Clamping function
void ControlneuromeshNode::transform_output(float* input, size_t size, float min_value, float max_value) {
    std::transform(input, input + size, input,
                   [min_value, max_value](float i) { 
                       return std::log(std::exp(std::clamp(i, min_value, max_value)) + 1.0) + 1.0; 
                   });

}

double ControlneuromeshNode::beta_distribution(double alpha, double beta, std::mt19937& gen) {
        std::gamma_distribution<> gamma_alpha(alpha, 1.0);
        std::gamma_distribution<> gamma_beta(beta, 1.0);

        double x = gamma_alpha(gen);
        double y = gamma_beta(gen);

        return x / (x + y);
}

void ControlneuromeshNode::run_decoder_cycle() {
    //RCLCPP_INFO(this->get_logger(), "Entering run_decoder_cycle");

    if (gnn_result_future.valid()) {
        std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> gnn_results = gnn_result_future.get();
        stopClock("decoder_inference");

        if (gnn_results.size() != 1) {
            RCLCPP_WARN(this->get_logger(), "Unexpected number of tensors in the inference result.");
            received_features_.clear();
            return;
        }

        auto gnn_result = gnn_results[0];
        if (gnn_result->result != 0) {
            RCLCPP_WARN(this->get_logger(), "Decoder inference failed.");
            received_features_.clear();
            return;
        }

        // Apply beta distribution and prepare Twist message
        geometry_msgs::msg::Twist twist_msg;
        std::vector<double> processed_values;
        float* alpha = reinterpret_cast<float*>(&gnn_result->data[0]);

        //for (auto data : &gnn_result->data)
        //{
        //RCLCPP_INFO_STREAM(this->get_logger(), "GNN Results: " << data); 
        //}

        size_t data_size = gnn_result->data.size() / sizeof(float);
        RCLCPP_INFO_STREAM(this->get_logger(), "GNN Result size: " << data_size);

        RCLCPP_INFO_STREAM(this->get_logger(), "alpha1" << *(alpha));
	RCLCPP_INFO_STREAM(this->get_logger(), "alpha2" << *(alpha+1));
	RCLCPP_INFO_STREAM(this->get_logger(), "alpha3" << *(alpha+2));
	RCLCPP_INFO_STREAM(this->get_logger(), "alpha4" << *(alpha+3));



        // Apply clamping to the vector
        transform_output(alpha, data_size, std::log(1.e-16), -std::log(1.e16));
        float* beta = alpha + data_size/2;

        for (size_t i = 0; i < data_size/2; ++i) {
            double processed_value = beta_distribution(alpha[i], beta[i], gen_);
            processed_values.push_back(processed_value);
        }

        // Assuming the first two values correspond to linear.x and angular.z
        if (processed_values.size() >= 2) {
            twist_msg.linear.x = processed_values[0];
            twist_msg.angular.z = processed_values[1];
        }

        // Publish the Twist message
        velocity_publisher_->publish(twist_msg);

        //RCLCPP_INFO(this->get_logger(), "Decoder cycle completed in %lims", times["decoder_inference"].first);
        received_features_.clear();
        feature_buffer_timestamp_.clear();
    }

    //RCLCPP_INFO(this->get_logger(), "received_features_.size() is %d", received_features_.size());

    if (!received_features_.empty()) {
        startClock("decoder_inference");
        //RCLCPP_INFO(this->get_logger(), "Running decoder on features.");

        // Combine features of other robots using build encoder
        neuromesh_interfaces::msg::Tensor aggregated_tensor;
        if (buildDecoderTensor(received_features_, aggregated_tensor)) {
            //RCLCPP_INFO(this->get_logger(), "aggregated_tensor size is %d", aggregated_tensor.data.size());
            gnn_result_future = performInference(decoder_model_name_, {aggregated_tensor});
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to build decoder tensor");
        }
    }

    //RCLCPP_INFO(this->get_logger(), "Exiting run_decoder_cycle");
}

void ControlneuromeshNode::startClock(std::string phase){
	int64_t now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	times[phase] =  {now_time, false};
}
void ControlneuromeshNode::stopClock(std::string phase){
	if (times[phase].second){
		return; // clock already stopped
	}
	int64_t start_time = times[phase].first;
	int64_t now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	
	times[phase] = {now_time - start_time, true};
}

int64_t ControlneuromeshNode::checkClock(std::string phase){
	if (times[phase].second){
		return times[phase].first; // clock already stopped
	}

	//stopclock calculations without saving
	int64_t start_time = times[phase].first;
	int64_t now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	return now_time - start_time;
}

//TODO generalize to a transpose function
//tested for use with integers
neuromesh_interfaces::msg::Tensor ControlneuromeshNode::convert_to_nchw(const neuromesh_interfaces::msg::Tensor& input){
	std::vector<uint8_t> new_data;
	const std::vector<uint8_t>& old_data = input.data;

	std::vector<uint64_t> strides = input.strides;
	std::vector<uint32_t> dims = input.shape.dims;
	unsigned long bytedepth = input.strides[0] / input.shape.dims[0];


	for(int i = 0; i < dims[2]; i++){
		for(int j = 0; j < dims[0]; j++){
			for(int k = 0; k < dims[1]; k++){
				new_data.push_back(old_data[ (j * strides[0]) + (k * strides[1]) + (i * strides[2]) ] );
			}
		}
	}

	neuromesh_interfaces::msg::Tensor new_tensor = std::move(input);
	new_tensor.data = new_data;
	new_tensor.shape.dims = {input.shape.dims[2], input.shape.dims[0], input.shape.dims[1]};
	new_tensor.strides = {dims[1] * dims[2] * bytedepth, dims[2] * bytedepth, bytedepth};
	
	return new_tensor;
}

float int_to_scaled_float(int i){ return static_cast<float>(i) / 255.0;}
}
