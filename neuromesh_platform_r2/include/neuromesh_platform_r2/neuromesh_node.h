#ifndef neuromesh_NODE_HEADER_H
#define neuromesh_NODE_HEADER_H

#include "rclcpp/rclcpp.hpp"

#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

#include "neuromesh_interfaces/msg/tensor.hpp" 
#include "neuromesh_interfaces/msg/feature.hpp" 

namespace neuromeshNode {
class neuromeshNode : public rclcpp::Node
{
    //FUNCTIONS
public:
    //Constructor
    neuromeshNode(const rclcpp::NodeOptions &options);

protected:
    /**
    * @brief This function is called whenever a new feature is received. It will add the feature to the feature buffer.
    *
    * @param msg The feature message.
    */
    void feature_callback(const neuromesh_interfaces::msg::Feature::SharedPtr msg);

    /**
    * @brief This function is called whenever a new image is received. It will convert the image to a tensor and run inference on it.
    *
    * @param msg The image message.
    */
    void camera_callback(const sensor_msgs::msg::Image::SharedPtr msg);

    /**
    * @brief Run inference on the received features, then publish the output. This gets run once per decoder_cycle_length_.
    */
    void process_features();

    /**
    * @brief Sends a tensor to the engine for inference.
    *
    * NOTE: This function is a placeholder and should be implemented by the user.
    *
    * @param model_name A string to indicate which model to use, as implemented by the user.
    * @param tensor The input to the model.
    * @return A future that will contain the output tensor.
    */
	virtual std::future<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> performInference(const std::string& model_name, const neuromesh_interfaces::msg::Tensor& tensor);


    /**
     * @brief Convert a tensor to a feature message.
     * @param tensor Input message to be converted
     * @return Return tensor in feature format
     */
	neuromesh_interfaces::msg::Feature buildFeatureMessage(const neuromesh_interfaces::msg::Tensor& tensor);

    /**
    * @brief Aggregates features from feature buffer into a single tensor.
    *
    * NOTE: This function is a placeholder and should be implemented by the user.
    *
    * @param buffer The buffer of features (typically this->feature_buffer_).
    * @return A single tensor containing all the features.
    */
	virtual neuromesh_interfaces::msg::Tensor buildDecoderTensor(std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr> buffer);

    /**
     * @brief Convert a ROS image message to a tensor.
     * @param msg Input message to be converted
     * @return Return image in tensor format
     */
    neuromesh_interfaces::msg::Tensor imageToTensor(const sensor_msgs::msg::Image::SharedPtr msg);

    /**
     * @brief Creates a feature subscription to another agent and adds it to the feature_subscriptions_ map.
     * @param subscription_map map to add the subscription to.
     * @param id ID of the agent to subscribe to
     * @param qos QOS settings to use for the subcription
     */
    void createSubscription(std::map<std::string, rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr>& subscription_map, std::string id, rclcpp::QoS qos);

    /**
     * @brief Deletes a feature subscription to another agents and removes it from the feature_subscriptions_ map.
     * @param subscription_map map to remove the subscription from.
     * @param id ID of the agent to remove 
     */
    void removeSubscription(std::map<std::string, rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr> subscription_map, std::string id);

    /**
     * @brief Takes a string given as a parameter to the node and converts it to a QoS profile.
     * 
     * NOTE: implementation taken from https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/nvblox_ros_common/src/qos.cpp#L26
     * 
     * @param str string corresponding to the QoS profile
     * @return QoS profile
     */
    rmw_qos_profile_t parseQoSString(const std::string& str);

    //Split agent string parameter into vector of agent ids
    /**
     * @brief Takes a string list of agents given as a parameter to the node and converts it to a set of agent ids.
     * @param str agent list string
     * @return set of agent ids
     */
    std::set<std::string> splitAgentString(std::string str);

    //Set encoder status as to whether or not it's already been run this cycle
    /**
     * @brief Takes the result of the encoder inference from camera_callback and publishes it as a feature message. Run every encoder_cycle_length_.
     */
    void run_encoder_cycle();

    /**
     * @brief Converts a tensor from NHWC to NCHW format. 
     * 
     * NOTE: Only tested for use with integers. Future work should include replacing this with a general transpose function that supports floats.
     * 
     * @param input Input NHWC tensor
     * @return output NCHW tensor
     */
    neuromesh_interfaces::msg::Tensor convert_to_nchw(const neuromesh_interfaces::msg::Tensor& input);


    /**
     * @brief Converts a tensor of ints to a tensor of float32s.
     * 
     * NOTE: Future work should include replacing this with a general function that supports all datatypes.
     * 
     * @param input input int tensor
     * @return output float tensor
     */
     //Converts a vector of unsinged integers from an image to floats
    neuromesh_interfaces::msg::Tensor tensor_ints_to_floats(neuromesh_interfaces::msg::Tensor& input);

    //***TIMING FUNCTIONS***
    /**
     * @brief Start timing a phase of the code.
     * 
     * NOTE: Multiple phases can be timed concurrently, but each timer overwrites the previous time and timer from its phase. 
     * 
     * @param phase denotes what the timer is timing
     */
    void startClock(std::string phase);
    /**
     * @brief stop timing a phase of the code.
     * 
     * NOTE: Multiple phases can be timed concurrently, but each timer overwrites the previous time and timer from its phase. 
     * 
     * @param phase denotes what the timer is timing
     */
    void stopClock(std::string phase);
    /**
     * @brief Check a timer, either running or stopped, and return the time in milliseconds since the timer was started.
     * @param phase denotes what the timer is timing
     * @return time in milliseconds since the timer was started
     */
    int64_t checkClock(std::string phase);

    /**
     * NOTE: The second item in the times pair indicates whether the timer is finished. If is it, then the first value represents the time in milliseconds, otherwise the first value represents the starting time.
     */
    std::map<std::string, std::pair<int64_t, bool>> times;

    //VARIABLES

    //Lists of agent ids that describe 1) all agents 2) the available ones
	std::set<std::string> all_agents;
	std::set<std::string> available_agents;

	//publishers and subscriptions
	rclcpp::Publisher<neuromesh_interfaces::msg::Feature>::SharedPtr feature_publisher_;
	rclcpp::Publisher<neuromesh_interfaces::msg::Tensor>::SharedPtr output_publisher_;
	std::map<std::string, rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr> feature_subscriptions_;
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_subscription_;

	//Timer functions to run encoder/decoder cycles
	rclcpp::TimerBase::SharedPtr decoder_timer_; //runs process_features
    rclcpp::TimerBase::SharedPtr encoder_timer_; //runs run_encoder_cycle

    //Denotes whether the encoder has already been run this cycle.
    bool fresh_encoder_cycle; 

    //Futures for the engine result
    std::future<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> encoder_result;
    std::future<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> gnn_result_future;

	//variables for features
	std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr> feature_buffer_;

	//parameters
	std::string encoder_model_name_;
	std::string decoder_model_name_;
	std::string topic_prefix_;
	std::string output_topic_;
	int decoder_cycle_length_;
    int encoder_cycle_length_;
    int encoder_await_length_;
	std::string id_;
    std::string image_qos_profile_;
    std::string features_qos_profile_;
    std::string output_qos_profile_;
    std::string agents_;
    bool to_nchw_;
    bool ints_to_floats_;

};
}

#endif //neuromesh_NODE_HEADER_H
