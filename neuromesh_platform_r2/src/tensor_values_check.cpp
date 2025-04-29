#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "sensor_msgs/image_encodings.hpp"
#include <neuromesh_interfaces/msg/tensor.hpp>
#include <memory>
#include <vector>
#include <cstring>
#include <cmath>

class TensorCheckNode : public rclcpp::Node
{
public:
    TensorCheckNode() : Node("tensor_check_node")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/race13/color/image_raw", 10,
            std::bind(&TensorCheckNode::imageCallback, this, std::placeholders::_1));
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        auto tensor = imageToTensor(msg);
        
        // Print some tensor values for verification
        RCLCPP_INFO(this->get_logger(), "Tensor shape: %dx%dx%dx%d",
                    tensor.shape.dims[0], tensor.shape.dims[1], tensor.shape.dims[2], tensor.shape.dims[3]);
        RCLCPP_INFO(this->get_logger(), "Tensor data type: %d", tensor.data_type);
        
        // Print first few values of the tensor
        const float* float_data = reinterpret_cast<const float*>(tensor.data.data());
        for (int i = 0; i < 10; ++i) {
            RCLCPP_INFO(this->get_logger(), "Tensor value %d: %f", i, float_data[i]);
        }
    }

    // Helper function for bicubic interpolation
    float cubicInterpolate(float p[4], float x) {
        return p[1] + 0.5 * x*(p[2] - p[0] + x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x*(3.0*(p[1] - p[2]) + p[3] - p[0])));
    }

    // Bicubic resizing function
    std::vector<float> bicubicResize(const std::vector<uint8_t>& input, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int channels) {
        std::vector<float> output(outputWidth * outputHeight * channels);
        float scaleX = static_cast<float>(inputWidth) / outputWidth;
        float scaleY = static_cast<float>(inputHeight) / outputHeight;

        for (int y = 0; y < outputHeight; ++y) {
            for (int x = 0; x < outputWidth; ++x) {
                float gx = x * scaleX;
                float gy = y * scaleY;
                int gxi = static_cast<int>(gx);
                int gyi = static_cast<int>(gy);

                for (int c = 0; c < channels; ++c) {
                    float p[4][4];
                    for (int yy = 0; yy < 4; ++yy) {
                        for (int xx = 0; xx < 4; ++xx) {
                            int xCoord = std::clamp(gxi + xx - 1, 0, inputWidth - 1);
                            int yCoord = std::clamp(gyi + yy - 1, 0, inputHeight - 1);
                            p[yy][xx] = input[(yCoord * inputWidth + xCoord) * channels + c];
                        }
                    }

                    float arr[4];
                    for (int i = 0; i < 4; ++i) {
                        arr[i] = cubicInterpolate(p[i], gx - gxi);
                    }

                    output[(y * outputWidth + x) * channels + c] = cubicInterpolate(arr, gy - gyi);
                }
            }
        }

        return output;
    }

    neuromesh_interfaces::msg::Tensor imageToTensor(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        neuromesh_interfaces::msg::Tensor tensor = neuromesh_interfaces::msg::Tensor();
        // Original dimensions
        uint channels = sensor_msgs::image_encodings::numChannels(msg->encoding);
        uint height = msg->height;
        uint width = msg->width;
		int size = 224;
		bool square_ok = false;
        
        // Resize
		uint new_height = 512;
        uint new_width = 384;

        // Implement bicubic interpolation for resizing
        std::vector<float> resized_data(new_width * new_height * channels);
        // For simplicity, we'll use nearest-neighbor interpolation here
        // Replace this with proper bicubic interpolation for better results
        for (uint y = 0; y < new_height; ++y) {
            for (uint x = 0; x < new_width; ++x) {
                uint orig_x = static_cast<uint>(x * width / new_width);
                uint orig_y = static_cast<uint>(y * height / new_height);
                for (uint c = 0; c < channels; ++c) {
                    resized_data[(y * new_width + x) * channels + c] = 
                        static_cast<float>(msg->data[(orig_y * width + orig_x) * channels + c]);
                }
            }
        }

        // Crop
        uint cx = new_width / 2;
        uint cy = new_height / 2;
        uint halfw, halfh;
        if (size == 224) {
            halfw = halfh = std::min(cx, cy);
        } else {
            halfw = ((2 * cx) / 16) * 8;
            halfh = ((2 * cy) / 16) * 8;
            if (!square_ok && new_width == new_height) {
                halfh = static_cast<uint>(3 * halfw / 4);
            }
        }
        uint crop_width = 2 * halfw;
        uint crop_height = 2 * halfh;
        std::vector<float> cropped_data(crop_width * crop_height * channels);
        for (uint y = 0; y < crop_height; ++y) {
            for (uint x = 0; x < crop_width; ++x) {
                uint src_x = cx - halfw + x;
                uint src_y = cy - halfh + y;
                for (uint c = 0; c < channels; ++c) {
                    cropped_data[(y * crop_width + x) * channels + c] = 
                        resized_data[(src_y * new_width + src_x) * channels + c];
                }
            }
        }

        // Normalize
        for (auto& pixel : cropped_data) {
            pixel = (pixel / 255.0f - 0.5f) / 0.5f;
        }

        // Set tensor data
        tensor.data.resize(crop_width * crop_height * channels * sizeof(float));
        std::memcpy(tensor.data.data(), cropped_data.data(), tensor.data.size());

        // Set datatype to float32
        tensor.data_type = 9; // float32

        // Set shape
        tensor.shape.dims = std::vector<uint32_t>{1, channels, crop_height, crop_width}; // NCHW format
        tensor.shape.rank = 4;
        
        // Set strides
        tensor.strides = std::vector<uint64_t>{
            channels * crop_height * crop_width * sizeof(float),
            crop_height * crop_width * sizeof(float),
            crop_width * sizeof(float),
            sizeof(float)
        };

        // Set name
        tensor.name = msg->header.frame_id + std::to_string(msg->header.stamp.sec) + "." + std::to_string(msg->header.stamp.nanosec);

        return tensor;
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TensorCheckNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}