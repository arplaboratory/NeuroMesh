#include "neuromesh_platform_r2/gat_planner_neuromesh_node.h"
#include "rclcpp/rclcpp.hpp"
#include <chrono>

namespace GATPlannerNeuromeshNode {
GATPlannerNeuromeshNode :: GATPlannerNeuromeshNode(const rclcpp::NodeOptions &options): Node("GAT_planner_neuromesh_node", options)
{
	// Declare node parameters
	this->declare_parameter<std::string>("encoder_model_name", "default_encoder_model");
	this->declare_parameter<std::string>("decoder_model1_name", "default_decoder_model");
    this->declare_parameter<std::string>("decoder_model2_name", "default_decoder_model");
	this->declare_parameter<std::string>("topic_prefix", "features_");
    this->declare_parameter<std::string>("pos_topic_prefix", "pos_");
	this->declare_parameter<std::string>("output_topic", "gnn_output_");
    this->declare_parameter<std::string>("planning_frame", "map");
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
    this->declare_parameter<std::string>("goal_poses_yaml_file", "goal_poses.yaml");
    this->declare_parameter<double>("goals_sending_delay", 10.0);

	// Get node parameters
	this->get_parameter("encoder_model_name", encoder_model_name_);
	this->get_parameter("decoder_model1_name", decoder_model1_name_);
    this->get_parameter("decoder_model2_name", decoder_model2_name_);
	this->get_parameter("topic_prefix", topic_prefix_);
    this->get_parameter("pos_topic_prefix", pos_topic_prefix_);
	this->get_parameter("output_topic", output_topic_);
    this->get_parameter("planning_frame", planning_frame_);
	this->get_parameter("decoder_cycle_length", decoder_cycle_length_);
	this->get_parameter("encoder_cycle_length", encoder_cycle_length_);
	this->get_parameter("encoder_await_length", encoder_await_length_);
	this->get_parameter("id", id_);
	this->get_parameter("image_qos_profile", image_qos_profile_);
	this->get_parameter("features_qos_profile", features_qos_profile_); //for both input and output
	this->get_parameter("output_qos_profile", output_qos_profile_);
	this->get_parameter("agents", agents_);
	this->get_parameter("to_nchw", to_nchw_);
    this->get_parameter("goal_poses_yaml_file", goal_poses_yaml_file);
    this->get_parameter("goals_sending_delay", goals_sending_delay_);

	auto feature_qos = rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(features_qos_profile_));
	auto output_qos = rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(output_qos_profile_));
	feature_publisher_ = this->create_publisher<neuromesh_interfaces::msg::Feature>(topic_prefix_ + id_, feature_qos);

    // Initialize GNN result publisher and 
    gnn_result_publisher_ = this->create_publisher<neuromesh_interfaces::msg::Feature>(
        output_topic_ + id_, feature_qos
    );

    second_decoder_result_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    "second_decoder_result_topic", 10  // topic name and queue size
    );

	//PLACEHOLDER: update available_agents
	
	all_agents = splitAgentString(agents_);
	for (const auto& agent : all_agents) {
		RCLCPP_DEBUG(this->get_logger(), "%s", agent.c_str());
	}
	all_agents.erase(id_); //remove self from list

    // Print out state vector elements
    neuromesh_interfaces::msg::StateVector tmp;
    for (uint i = 0; i < 6; ++i)
    {
        tmp.state_vector.push_back(0.);
    }
    current_states_.insert(std::pair(id_, tmp));

    // for (const auto& id : all_agents) {
    //     current_states_.emplace(id, tmp);
    // }

	available_agents = all_agents; 

    // Print out all agent names in the agent list
	for (std::string id : all_agents){
		RCLCPP_DEBUG(this->get_logger(), "Going through all agents to create subscriptions");
		RCLCPP_DEBUG(this->get_logger(), "Id: %s", id.c_str());
		this->createSubscription(feature_subscriptions_, id, feature_qos);
        this->createPosSubscription(pos_subscriptions_, id, feature_qos);
        this->createGNNSubscription(gnn_subscriptions_, id, feature_qos);
	}

    // Extra print statements for debugging
    for(const auto& [key, value] : current_states_) {
        RCLCPP_DEBUG(
            get_logger(), 
            "Key: %s, State Vector Size: %zu", 
            key.c_str(), 
            value.state_vector.size()
        );
    }

    // Load the goal poses for the robots from the yaml file
    load_goal_poses_from_yaml();

    // Create timers for encoder and decoder cycles
	decoder_timer_ = this->create_wall_timer(std::chrono::duration<int,std::milli>(decoder_cycle_length_), std::bind(&GATPlannerNeuromeshNode::run_decoder_cycle, this));
	encoder_timer_ = this->create_wall_timer(std::chrono::duration<int,std::milli>(encoder_cycle_length_), std::bind(&GATPlannerNeuromeshNode::run_encoder_cycle, this));
	fresh_encoder_cycle = true;
	waypoint_cmd_sent_ = false;

    // Initialize tf buffers
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
}

void GATPlannerNeuromeshNode::load_goal_poses_from_yaml() {
    try {
        // Clear existing goal poses
        goal_poses_.clear();
        
        // Load the YAML file
        YAML::Node config = YAML::LoadFile(goal_poses_yaml_file);
        
        RCLCPP_DEBUG(this->get_logger(), "Loading goals for all robots");
        
        // Iterate through all top-level nodes (robot names)
        for (const auto& robot_entry : config) {
            std::string robot_name = robot_entry.first.as<std::string>();
            
            // Check if this robot has a goals section
            if (!robot_entry.second["goals"] || !robot_entry.second["goals"].IsSequence()) {
                RCLCPP_WARN(this->get_logger(), "No valid goals found for robot '%s'", robot_name.c_str());
                continue;
            }
            
            // Process each goal for this robot
            for (const auto& goal : robot_entry.second["goals"]) {
                if (!goal["position"] || !goal["position"]["x"] || !goal["position"]["y"]) {
                    RCLCPP_WARN(this->get_logger(), "Skipping malformed goal entry for robot '%s'", robot_name.c_str());
                    continue;
                }
                
                geometry_msgs::msg::Pose pose;
                pose.position.x = goal["position"]["x"].as<double>();
                pose.position.y = goal["position"]["y"].as<double>();
                pose.position.z = 0.0;  // Set to 0 if not needed
                
                goal_poses_.push_back(pose);
                
                RCLCPP_DEBUG(
                    this->get_logger(),
                    "Loaded goal for robot '%s': Position (%.2f, %.2f)",
                    robot_name.c_str(),
                    pose.position.x,
                    pose.position.y
                );
            }
        }

        RCLCPP_DEBUG(
            this->get_logger(),
            "Successfully loaded %zu goal poses for all robots",
            goal_poses_.size()
        );
        
    } catch (const YAML::Exception& e) {
        RCLCPP_ERROR(
            this->get_logger(),
            "Failed to load goals from YAML file: %s",
            e.what()
        );
    }
}

void GATPlannerNeuromeshNode::pos_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {

    // Parse position data
    pos_callback_complete = true;

    current_states_[id_].state_vector[0] = msg->pose.pose.position.x;
    current_states_[id_].state_vector[1] = msg->pose.pose.position.y;

    // Set orientation to [0, 0, 0, 1]
    current_states_[id_].state_vector[2] = 0.0;
    current_states_[id_].state_vector[3] = 0.0;
    current_states_[id_].state_vector[4] = 0.0;
    current_states_[id_].state_vector[5] = 1.0;

    // Keep spamming others until goal is sent
    if(!waypoint_cmd_sent_) {
        run_encoder_cycle();
    }

}

void GATPlannerNeuromeshNode::neighbor_pos_callback(const nav_msgs::msg::Odometry::SharedPtr msg, const std::string &id) {
    // if empty key, initialize state_vector
    auto& state = current_states_[id];
    if (state.state_vector.size() != 6) {
        state.state_vector.resize(6, 0.0);
    }

    current_states_[id].state_vector[0] = msg->pose.pose.position.x;
    current_states_[id].state_vector[1] = msg->pose.pose.position.y;

    // Set orientation to [0, 0, 0, 1]
    current_states_[id].state_vector[2] = 0.0;
    current_states_[id].state_vector[3] = 0.0;
    current_states_[id].state_vector[4] = 0.0;
    current_states_[id].state_vector[5] = 1.0;
}

neuromesh_interfaces::msg::Tensor GATPlannerNeuromeshNode::calculate_input_features(const std::vector<float>& state_vector, const std::vector<std::string> neighbors_ids) {
    neuromesh_interfaces::msg::Tensor tensor_msg;
    // Set tensor dimensions
    tensor_msg.shape.dims = {1, static_cast<int64_t>((1 + goal_poses_.size() + 2)*2)}; // ego robot pos + relative pos to 4 goals + relative pos to 2 neighbors

    tensor_msg.data_type = 9; // float32
    
    std::vector<float> input_feature;
    input_feature.push_back(state_vector[0] / 2.0f);
    input_feature.push_back(state_vector[1] / 2.0f);

    // Relative pos to the goals
    for (size_t j = 0; j < goal_poses_.size(); ++j) {
        float goal_x = goal_poses_[j].position.x;
        float goal_y = goal_poses_[j].position.y;
    
        float dx = (goal_x - state_vector[0]) / 4.0f; // hard coded normalization by 2*env_bound
        float dy = (goal_y - state_vector[1]) / 4.0f;

        input_feature.push_back(dx);
        input_feature.push_back(dy);
    }
    // Relative pos to the robots
    for (size_t n = 0; n < neighbors_ids.size(); ++n) {\
        float neighbor_x = current_states_[neighbors_ids[n]].state_vector[0];
        float neighbor_y = current_states_[neighbors_ids[n]].state_vector[1];
    
        float dx = (neighbor_x - state_vector[0]) / 4.0f; // hard coded normalization by 2*env_bound
        float dy = (neighbor_y - state_vector[1]) / 4.0f;
    
        // Write into input_feat starting at index 2
        input_feature.push_back(dx);
        input_feature.push_back(dy);
    }

    // Directly copy float values to tensor data
    tensor_msg.data.resize(input_feature.size() * sizeof(float));
    std::copy(
        reinterpret_cast<unsigned char*>(input_feature.data()),
        reinterpret_cast<unsigned char*>(input_feature.data() + input_feature.size()),
        tensor_msg.data.begin()
    );
    
    // Set data type explicitly
    tensor_msg.data_type = 9;

    // Set shape explicitly
    tensor_msg.shape.dims = {1, static_cast<int>(input_feature.size())};

    // Optionally set strides
    tensor_msg.strides = {static_cast<int64_t>(input_feature.size()), 1};

    return tensor_msg;
}

std::vector<std::string> GATPlannerNeuromeshNode::get_closest_neighbors() {
    const auto& ego_state = current_states_[id_];
    double ego_x = ego_state.state_vector[0];
    double ego_y = ego_state.state_vector[1];

    std::vector<std::pair<std::string, double>> distances;
    for (const auto& [agent_id, state] : current_states_) {
        if (agent_id == id_) continue;  // skip ego
    
        double dx = state.state_vector[0] - ego_x;
        double dy = state.state_vector[1] - ego_y;
        double dist = std::sqrt(dx * dx + dy * dy);
    
        distances.emplace_back(agent_id, dist);
    }
    std::sort(distances.begin(), distances.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    std::vector<std::string> closest_ids;
        for (size_t i = 0; i < std::min(size_t(2), distances.size()); ++i) {
            closest_ids.push_back(distances[i].first);
        }
    return closest_ids;
}

neuromesh_interfaces::msg::Feature GATPlannerNeuromeshNode::buildFeatureMessage(const neuromesh_interfaces::msg::Tensor& tensor)
{
	neuromesh_interfaces::msg::Feature feature_msg = neuromesh_interfaces::msg::Feature();

	feature_msg.tensor = tensor;
	feature_msg.id = id_;
	feature_msg.timestamp = this->get_clock()->now();
	return feature_msg;
}

void GATPlannerNeuromeshNode::feature_callback(const neuromesh_interfaces::msg::Feature::SharedPtr msg) {
    std::string uuid = msg->id;
    received_features_[uuid] = msg;
    feature_buffer_timestamp_[uuid] = msg->timestamp.sec + msg->timestamp.nanosec*1e-9;
    auto feature_now = this->get_clock()->now();
    double feature_timestamp = feature_now.seconds() + feature_now.nanoseconds() / 1e9;  
    RCLCPP_DEBUG(this->get_logger(), "Delay between receiving and publishing features %.9f", feature_timestamp - feature_buffer_timestamp_[uuid]);   
    
    // std::cout << "Delay between receiving and publishing features " << feature_now.seconds() - feature_buffer_timestamp_[uuid] << " seconds" << std::endl
    //           << "feature_buff_timestamp_[uuid]: " << feature_buffer_timestamp_[uuid] << std::endl
    //           << "msg->timestamp.sec: " << msg->timestamp.sec << std::endl
    //           << "msg->timestamp.nanosec: " << msg->timestamp.nanosec << std::endl
    //           << "msg->timestamp.nanosec*1e-9: " << msg->timestamp.nanosec*1e-9 << std::endl
    //         //   << "feature_timestamp: " << feature_timestamp << std::endl
    //           << "feature_now.seconds(): " << feature_now.seconds() << std::endl
    //           << "feature_now.nanoseconds(): " << feature_now.nanoseconds() << std::endl
    //           << "feature_now.nanoseconds() / 1e9: " << feature_now.nanoseconds() / 1e9 << std::endl
    //           << "Clock type: " << this->get_clock()->get_clock_type() << std::endl;
}

//perform inference on tensor using model called model_name
//PLACEHOLDER
std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> GATPlannerNeuromeshNode::performInference(const std::string& model_name, const std::vector<neuromesh_interfaces::msg::Tensor>& tensors)
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
bool GATPlannerNeuromeshNode::buildDecoderTensor(
        std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr>& agent_features,
        neuromesh_interfaces::msg::Tensor& own_feature,
        neuromesh_interfaces::msg::Tensor& aggregated_tensor)
{
	return true;
}

//Adds a subscription to the feature_subscriptions_ map
void GATPlannerNeuromeshNode::createSubscription(std::map<std::string, rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr>& subscription_map, std::string id, rclcpp::QoS qos)
{
	std::string topic = topic_prefix_ + id;

	RCLCPP_DEBUG(this->get_logger(), "creating subscription for topic %s", topic.c_str());

	rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr feature_subscription_ = 
		this->create_subscription<neuromesh_interfaces::msg::Feature>(
			topic,
			qos,
			std::bind(&GATPlannerNeuromeshNode::feature_callback, this, std::placeholders::_1));

	subscription_map.insert( {id, feature_subscription_} );
}

void GATPlannerNeuromeshNode::createPosSubscription(std::map<std::string, rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr>& pos_subscription_map, std::string id, rclcpp::QoS qos)
{
	std::string topic = pos_topic_prefix_ + id;

	RCLCPP_DEBUG(this->get_logger(), "creating subscription for topic %s", topic.c_str());

    auto lambda_callback = [this, id](const nav_msgs::msg::Odometry::SharedPtr msg) {
        this->neighbor_pos_callback(msg, id);
    };

	rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pos_subscription_ = 
		this->create_subscription<nav_msgs::msg::Odometry>(
			topic,
			qos,
			lambda_callback);

	pos_subscription_map.insert( {id, pos_subscription_} );
}

//Remove subscription form the feature_subscriptions_ map
void GATPlannerNeuromeshNode::removeSubscription(std::map<std::string, rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr> subscription_map, std::string id)
{
	subscription_map.erase(id);
}

//Adds a subscription to the gnn_subscriptions_ map
void GATPlannerNeuromeshNode::createGNNSubscription(std::map<std::string, rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr>& gnn_subscription_map, std::string id, rclcpp::QoS qos)
{
	std::string topic = output_topic_ + id;

	RCLCPP_DEBUG(this->get_logger(), "creating gnn subscription for topic %s", topic.c_str());

	rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr gnn_subscription_ = 
		this->create_subscription<neuromesh_interfaces::msg::Feature>(
			topic,
			qos,
			std::bind(&GATPlannerNeuromeshNode::gnn_result_callback, this, std::placeholders::_1));

	gnn_subscription_map.insert( {id, gnn_subscription_} );
}

//Remove subscription form the gnn_subscriptions_ map
void GATPlannerNeuromeshNode::removeGNNSubscription(std::map<std::string, rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr> gnn_subscription_map, std::string id)
{
	gnn_subscription_map.erase(id);
}

//Convert string to ROS2 QoS profile
//from https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/nvblox_ros_common/src/qos.cpp#L26
rmw_qos_profile_t GATPlannerNeuromeshNode::parseQoSString(const std::string& str)
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
std::set<std::string> GATPlannerNeuromeshNode::splitAgentString(std::string str)
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

void GATPlannerNeuromeshNode::run_encoder_cycle() {
    // Run the encoder on all feature vectors
    // RCLCPP_INFO(this->get_logger(), "current_states_ size = %d", current_states_.size());
    if (fresh_encoder_cycle && current_states_.size()==4 && pos_callback_complete && !goal_poses_.empty() && !waypoint_cmd_sent_) {
        fresh_encoder_cycle = false;
        startClock("encoder_inference");

        // TO CHANGE
        std::vector<std::string> closest_neighbors_ids = get_closest_neighbors();
        neuromesh_interfaces::msg::Tensor input_features = calculate_input_features(
            std::vector<float>(current_states_[id_].state_vector.begin(), 
                            current_states_[id_].state_vector.end()), closest_neighbors_ids
        );
        size_t num_floats = input_features.data.size() / sizeof(float);
        const float* float_data = reinterpret_cast<const float*>(input_features.data.data());

        std::ostringstream oss;
        oss << "input_features: [";
        for (size_t i = 0; i < num_floats; ++i) {
            oss << float_data[i];
            if (i < num_floats - 1)
                oss << ", ";
        }
        oss << "]";

        // RCLCPP_INFO(this->get_logger(), "input features = %s", oss.str().c_str());

        auto encoder_now = this->get_clock()->now();
        // RCLCPP_INFO(this->get_logger(), "Performing inference");
        encoder_result = performInference(encoder_model_name_, {input_features});
        auto encoder_end = this->get_clock()->now();

        RCLCPP_DEBUG(this->get_logger(), "Finished performing Inference on Encoder");
    }

    // Keep publishing features to others
    if (!fresh_encoder_cycle && encoder_result.valid()) {
        RCLCPP_INFO(this->get_logger(), "I am here");
        auto encoder_status = encoder_result.wait_for(std::chrono::milliseconds(0));
        if (encoder_status == std::future_status::ready) {
            stopClock("encoder_inference");
            std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> encoded_features = encoder_result.get();

            std::shared_ptr<neuromesh_interfaces::msg::Tensor> feature_tensor = encoded_features[0];

            // Print out encoder model output
            float max_prob, next;
            size_t size = feature_tensor->data.size();
            std::memcpy(&max_prob, &feature_tensor->data[0], sizeof(float));
            std::stringstream ss;
            ss << "[" << std::fixed << std::setprecision(2) << max_prob;
            for (size_t i = 4; i < size; i +=4) {
                if (i + 3 < size)
                {
                    std::memcpy(&next, &feature_tensor->data[i], sizeof(float));
                    ss << " " << next << " ";
                    if (((i/4) % 5) == 0 ){
                        ss << "\n";
                    }
                }
            }
            ss << "]";

            RCLCPP_INFO(this->get_logger(), "Encoder features output:\n Size: %zu\n Output: %s", size, ss.str().c_str());
                                                
            neuromesh_interfaces::msg::Feature feature_msg = buildFeatureMessage(*feature_tensor.get());
            
            // Publish features
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), 
                    *this->get_clock(), 
                    1000, 
                    "Publishing features.");
            feature_publisher_->publish(feature_msg);

            // Store features
            received_features_[this->id_] = std::make_shared<neuromesh_interfaces::msg::Feature>(feature_msg);  

            
            feature_buffer_timestamp_[this->id_] = feature_msg.timestamp.sec + feature_msg.timestamp.nanosec*1e-9;

            fresh_encoder_cycle = true;
            RCLCPP_DEBUG(this->get_logger(), "Encoder cycle completed");

        } else if (checkClock("encoder_inference") >= encoder_await_length_) {
            RCLCPP_WARN(this->get_logger(), "Encoder inference timed out");
            fresh_encoder_cycle = true;
        }
    }
}

void GATPlannerNeuromeshNode::run_decoder_cycle() {
    // Handle previous decoder inference result
    if (gnn_result_future.valid()) {
        auto decoder_status = gnn_result_future.wait_for(std::chrono::milliseconds(0));
        if (decoder_status == std::future_status::ready) {
            stopClock("decoder_inference");
            std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> gnn_result = gnn_result_future.get();

            std::shared_ptr<neuromesh_interfaces::msg::Tensor> gnn_result_tensor = gnn_result[0];

            // Print first decoder output values
            float max_prob, next;
            size_t size = gnn_result_tensor->data.size();
            std::memcpy(&max_prob, &gnn_result_tensor->data[0], sizeof(float));
            std::stringstream ss;
            ss << "[" << std::fixed << std::setprecision(2) << max_prob;
            for (size_t i = 4; i < size; i +=4) {
                if (i + 3 < size)
                {
                    std::memcpy(&next, &gnn_result_tensor->data[i], sizeof(float));
                    ss << " " << next << " ";
                    if (((i/4) % 5) == 0 ){
                        ss << "\n";
                    }
                }
            }
            ss << "]";

            const auto& data = gnn_result_tensor->data;

            // Publish the GNN result to other robots
            neuromesh_interfaces::msg::Feature gnn_msg = build_gnn_msg(*gnn_result_tensor.get());

            // Publish the result
            gnn_result_publisher_->publish(gnn_msg);

            RCLCPP_DEBUG(this->get_logger(), "First Decoder inference completed and published");
            
            // Store local GNN result
            received_gnn_results_[this->id_] = std::make_shared<neuromesh_interfaces::msg::Feature>(gnn_msg);
            
            // Prepare for potential second-stage decoding
            prepare_second_stage_decoding();

            RCLCPP_DEBUG(this->get_logger(), "Second Decoder inference completed and published");
        }
    }

    // Check if we have enough feature messages (5 total)
    if (!received_features_.empty() && !waypoint_cmd_sent_) {
        startClock("decoder_inference");

        // Step 4: Combine features of other robots
        // Prepare encoder output tensor (from previous encoder cycle)
        encoder_output_tensor;
        neuromesh_interfaces::msg::Tensor aggregated_tensor;
        if (buildDecoderTensor(received_features_, encoder_output_tensor, aggregated_tensor)) {
            std::vector<neuromesh_interfaces::msg::Tensor> decoder_tensors = {encoder_output_tensor, aggregated_tensor, encoder_output_tensor};
            auto first_gnn_start = this->get_clock()->now();
            gnn_result_future = performInference(decoder_model1_name_, decoder_tensors);
            auto first_gnn_end = this->get_clock()->now();
            //std::cout <<"Computation time first gnn round" <<first_gnn_end.seconds() - first_gnn_start.seconds()<<std::endl;
            // double gnn_timestamp = now.seconds() + now.nanoseconds() / 1e9;

            // for (const auto& entry : feature_buffer_timestamp_) {
            //     const std::string& robot_name = entry.first;
            //     const double timestamp = entry.second;
            //     std::cout << "Robot " << robot_name << " Delay between first inference end and message publication" <<first_gnn_end.seconds() - timestamp <<std::endl;
            // }

        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to build decoder tensor");
        }

        // Clear received features for the next cycle
        received_features_.clear();
        feature_buffer_timestamp_.clear();
    }
}

neuromesh_interfaces::msg::Feature GATPlannerNeuromeshNode::build_gnn_msg(const neuromesh_interfaces::msg::Tensor& gnn_result) {
    // Convert GNN result to Tensor message
    neuromesh_interfaces::msg::Feature result_msg = neuromesh_interfaces::msg::Feature();
    result_msg.tensor = gnn_result;
    
    // Add metadata if needed
    result_msg.id = this->id_;
    result_msg.timestamp = this->get_clock()->now();

    return result_msg;
}

void GATPlannerNeuromeshNode::gnn_result_callback(const neuromesh_interfaces::msg::Feature::SharedPtr msg) {
    // Store received GNN result
    std::string gnn_id = msg->id;
    received_gnn_results_[gnn_id] = msg; 
    // Optional: Log received result
    RCLCPP_DEBUG(this->get_logger(),"Received GNN result from robot %d", msg->id);
}

void GATPlannerNeuromeshNode::prepare_second_stage_decoding() {

     if (second_decoder_result_future.valid()) {
        auto second_decoder_status = second_decoder_result_future.wait_for(std::chrono::milliseconds(0));
        if (second_decoder_status == std::future_status::ready) {
            stopClock("decoder_inference");
            std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> second_decoder_result = second_decoder_result_future.get();

            // Ensure the vector is not empty
            if (!second_decoder_result.empty()) {
                // Get the first tensor
                std::shared_ptr<neuromesh_interfaces::msg::Tensor> second_decoder_result_tensor = second_decoder_result[0];

                // Find the index of the maximum probability
                int max_prob_index = 0;

                // Final step to get decoder output, print the values and send it as waypoint goals to robots
                // Assuming the Tensor has a data field that is a vector of floats
                if (!second_decoder_result_tensor->data.empty()) {
                    float max_prob, next;
                    size_t size = second_decoder_result_tensor->data.size();
                    std::memcpy(&max_prob, &second_decoder_result_tensor->data[0], sizeof(float));
                    
                    // std::stringstream ss;
                    // ss << "[" << std::fixed << std::setprecision(2) << max_prob;
                    // for (size_t i = 4; i < size; i +=4) {
                    //     if (i + 3 < size)
                    //     {
                    //         std::memcpy(&next, &second_decoder_result_tensor->data[i], sizeof(float));
                    //         ss << " " << next << " ";
                    //         if (((i/4) % 5) == 0 ){
                    //             ss << "\n";
                    //         }
                    //         if (next > max_prob) {
                    //             max_prob = next;
                    //             max_prob_index = i/4;
                    //         }
                    //     }
                    // }
                    // ss << "]";
                    // RCLCPP_INFO_STREAM(this->get_logger(), "Second Decoder final output: " << second_decoder_result_tensor->data_type 
                    //                                         << "\n Size: " << size
                    //                                         << "\n Output: " << ss.str());

                    // Create and publish PoseStamped message
                    geometry_msgs::msg::PoseStamped pose_msg;
                    pose_msg.header.stamp = this->now();
                    pose_msg.header.frame_id = "map";  // Adjust frame_id as needed

                    pose_msg.pose = goal_poses_[max_prob_index];

                    RCLCPP_DEBUG(this->get_logger(), "Second Decoder inference completed and published");

                    YAML::Node yaml_string;
                    yaml_string["version"] = 2.0;
                    yaml_string["frame_id"] = planning_frame_;

                    // Create a waypoints sequence node
                    yaml_string["waypoints"] = YAML::Node(YAML::NodeType::Sequence);

                    // Create waypoint node
                    YAML::Node wp_node;
                    std::vector<float> pose_data{ 
                        pose_msg.pose.position.x, 
                        pose_msg.pose.position.y, 
                        pose_msg.pose.position.z,
                    };

                    wp_node["name"] = "waypoint1";
                    wp_node["pose"] = pose_data;
                    wp_node["pose"].SetStyle(YAML::EmitterStyle::Flow);
                    wp_node["radius"] = 2.0;

                    // Add waypoint to the sequence
                    yaml_string["waypoints"].push_back(wp_node);

                    std::stringstream pose_string;
                    pose_string << yaml_string;
                    
                }
            }
        }
     }

    // Check if we have received GNN results from all robots
    if (!received_gnn_results_.empty()) {
        // Prepare aggregated GNN results for second-stage decoder
        neuromesh_interfaces::msg::Tensor gnn_output_tensor;
        neuromesh_interfaces::msg::Tensor aggregated_gnn_output_tensor;
        if (buildDecoderTensor(received_gnn_results_, gnn_output_tensor, aggregated_gnn_output_tensor)) {

            second_decoder_result_future = performInference(decoder_model2_name_, {gnn_output_tensor, aggregated_gnn_output_tensor, encoder_output_tensor});

            auto gnn_now = this->get_clock()->now();

            for (const auto& entry : feature_buffer_timestamp_) {
                const std::string& robot_name = entry.first;
                double timestamp = entry.second;
            }

        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to build decoder tensor");
        }

        // Clear received GNN results for next cycle
        received_gnn_results_.clear();
    }
}

void GATPlannerNeuromeshNode::startClock(std::string phase){
	int64_t now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	times[phase] =  {now_time, false};
}
void GATPlannerNeuromeshNode::stopClock(std::string phase){
	if (times[phase].second){
		return; // clock already stopped
	}
	int64_t start_time = times[phase].first;
	int64_t now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	
	times[phase] = {now_time - start_time, true};
}

int64_t GATPlannerNeuromeshNode::checkClock(std::string phase){
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
neuromesh_interfaces::msg::Tensor GATPlannerNeuromeshNode::convert_to_nchw(const neuromesh_interfaces::msg::Tensor& input){
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
