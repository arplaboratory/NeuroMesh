#pragma once

#include <string>

struct NavigationGoal {
  double x{0.0};
  double y{0.0};
  double z{0.0};
  std::string planning_frame{"map"};
  std::string robot_id;
};