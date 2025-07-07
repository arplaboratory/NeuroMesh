#ifndef ENGINE_INTERFACE_TEST_FIXTURE_HPP
#define ENGINE_INTERFACE_TEST_FIXTURE_HPP

#include "mock_inference_engine.hpp"

class EngineInterfaceTest : public ::testing::Test {
    protected:
        engine_interface::MockInferenceEngine mock_engine;
    };

#endif // ENGINE_INTERFACE_TEST_FIXTURE_HPP