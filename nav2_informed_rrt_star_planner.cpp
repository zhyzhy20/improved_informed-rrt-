#include "nav2_custom_planner/nav2_informed_rrt_star_planner.hpp"
#include "nav2_core/exceptions.hpp"
#include "pluginlib/class_list_macros.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace nav2_custom_planner
{

nav_msgs::msg::Path InformedRRTStar::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  setStatsLabelOnce("InformedRRTStar");
  nav_msgs::msg::Path global_path;
  global_path.header.stamp = node_->now();
  global_path.header.frame_id = global_frame_;
  global_path.poses.clear();

  if (start.header.frame_id != global_frame_ || goal.header.frame_id != global_frame_) {
    RCLCPP_ERROR(node_->get_logger(), "坐标系不匹配！");
    clearMarkers();
    return global_path;
  }

  clearMarkers();

  const Node start_node(start.pose.position.x, start.pose.position.y);
  const Node goal_node(goal.pose.position.x, goal.pose.position.y);
  if (!isPointSafe(start_node.x, start_node.y) || !isPointSafe(goal_node.x, goal_node.y)) {
    RCLCPP_ERROR(node_->get_logger(), "起点或终点位于不可通行区域");
    throw nav2_core::PlannerException("Start or goal in collision");
  }

  const double start_goal_dist = distance(start_node, goal_node);
  if (start_goal_dist < 0.3 && isPathCollisionFree(start_node, goal_node)) {
    geometry_msgs::msg::PoseStamped start_pose = start;
    geometry_msgs::msg::PoseStamped goal_pose = goal;
    start_pose.header.frame_id = global_frame_;
    goal_pose.header.frame_id = global_frame_;
    start_pose.header.stamp = node_->now();
    goal_pose.header.stamp = start_pose.header.stamp;
    global_path.poses = {start_pose, goal_pose};
    return global_path;
  }

  const double min_x = costmap_->getOriginX();
  const double min_y = costmap_->getOriginY();
  const double max_x = min_x + costmap_->getSizeInMetersX();
  const double max_y = min_y + costmap_->getSizeInMetersY();
  std::uniform_real_distribution<double> rand_x(min_x, max_x);
  std::uniform_real_distribution<double> rand_y(min_y, max_y);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  std::vector<Node> tree{start_node};

  const double c_min = start_goal_dist;
  double best_cost = std::numeric_limits<double>::max();  // c_best
  int best_goal_parent_idx = -1;

  const auto planning_deadline =
    node_->now() + rclcpp::Duration::from_seconds(std::max(0.01, max_planning_time_sec_));

  auto sampleInformed = [&](const Node & s, const Node & g) -> Node {
    // Before we have a solution, do uniform sampling.
    if (best_cost == std::numeric_limits<double>::max() || best_cost <= c_min || c_min < 1e-3) {
      for (int retry = 0; retry < max_sampling_retry_; ++retry) {
        Node sample(rand_x(rand_gen_), rand_y(rand_gen_));
        if (isPointSafe(sample.x, sample.y)) {
          return sample;
        }
      }
      return g;
    }

    // Prolate ellipse informed sampling (no soft repulsion).
    const double a = best_cost / 2.0;
    const double b_sq = std::max(best_cost * best_cost - c_min * c_min, 1e-8);
    const double b = std::sqrt(b_sq) / 2.0;

    const double theta = std::atan2(g.y - s.y, g.x - s.x);
    for (int retry = 0; retry < max_sampling_retry_; ++retry) {
      const double r = std::sqrt(unit(rand_gen_));
      const double t = 2.0 * M_PI * unit(rand_gen_);
      const double x = a * r * std::cos(t);
      const double y = b * r * std::sin(t);

      // Rotate and translate
      const double xr = x * std::cos(theta) - y * std::sin(theta) + (s.x + g.x) / 2.0;
      const double yr = x * std::sin(theta) + y * std::cos(theta) + (s.y + g.y) / 2.0;

      const double x_clamped = std::max(min_x, std::min(xr, max_x));
      const double y_clamped = std::max(min_y, std::min(yr, max_y));
      const Node sample(x_clamped, y_clamped);
      if (isPointSafe(sample.x, sample.y)) {
        return sample;
      }
    }

    return g;
  };

  for (int iter = 0; iter < max_iterations_; ++iter) {
    if (node_->now() > planning_deadline) {
      break;
    }

    const bool sample_goal = unit_rand_(rand_gen_) < goal_sample_rate_;
    Node sample = sample_goal ? goal_node : sampleInformed(start_node, goal_node);
    if (!isPointSafe(sample.x, sample.y)) {
      continue;
    }
    drawSamplePoint(sample.x, sample.y);

    const int nearest_idx = findNearestNode(tree, sample.x, sample.y);
    const Node & nearest_node = tree[nearest_idx];
    const double dist = distance(nearest_node, sample);
    if (dist < 1e-6) {
      continue;
    }

    const double step = std::min(step_size_, dist);
    const Node steered(
      nearest_node.x + step * (sample.x - nearest_node.x) / dist,
      nearest_node.y + step * (sample.y - nearest_node.y) / dist);

    const double steer_cost = edgeTraversalCost(nearest_node, steered);
    Node new_node(steered.x, steered.y, nearest_idx, nearest_node.cost + steer_cost);

    if (!isPointSafe(new_node.x, new_node.y) || !isPathCollisionFree(nearest_node, new_node) ||
      new_node.cost == std::numeric_limits<double>::max() || !std::isfinite(new_node.cost))
    {
      continue;
    }

    const double dynamic_radius = std::min(search_radius_, computeRewireRadius(tree.size() + 1));
    const std::vector<int> near_nodes = findNearNodes(tree, new_node.x, new_node.y, dynamic_radius);

    // Best-parent selection.
    for (const int idx : near_nodes) {
      const double edge_cost = edgeTraversalCost(tree[idx], new_node);
      const double new_cost = tree[idx].cost + edge_cost;
      if (isPathCollisionFree(tree[idx], new_node) &&
        edge_cost < std::numeric_limits<double>::max() &&
        new_cost < new_node.cost)
      {
        new_node.parent_index = idx;
        new_node.cost = new_cost;
      }
    }

    tree.push_back(new_node);
    const int new_idx = static_cast<int>(tree.size()) - 1;
    recordNewTreeNodeTurn(tree, new_idx);

    // Rewiring
    for (const int idx : near_nodes) {
      const double edge_cost = edgeTraversalCost(new_node, tree[idx]);
      const double new_cost = new_node.cost + edge_cost;
      if (new_cost < tree[idx].cost &&
        edge_cost < std::numeric_limits<double>::max() &&
        isPathCollisionFree(new_node, tree[idx]))
      {
        tree[idx].parent_index = new_idx;
        tree[idx].cost = new_cost;
        updateDescendantCosts(tree, idx);
      }
    }

    drawTree(tree);

    if (distance(new_node, goal_node) < goal_tolerance_ && isPathCollisionFree(new_node, goal_node)) {
      const double goal_cost = new_node.cost + edgeTraversalCost(new_node, goal_node);
      if (goal_cost < best_cost) {
        best_cost = goal_cost;
        best_goal_parent_idx = new_idx;
      }
    }
  }

  if (best_goal_parent_idx == -1) {
    std::vector<geometry_msgs::msg::PoseStamped> fallback_path;
    if (createGridAStarFallback(start, goal, &fallback_path)) {
      global_path.poses = std::move(fallback_path);
      RCLCPP_WARN(node_->get_logger(), "Informed RRT* failed, fallback A* succeeded");
      return global_path;
    }
    throw nav2_core::PlannerException("Informed RRT* 规划失败");
  }

  tree.emplace_back(goal_node.x, goal_node.y, best_goal_parent_idx, best_cost);
  const int best_goal_idx = static_cast<int>(tree.size()) - 1;

  std::vector<geometry_msgs::msg::PoseStamped> raw_path = backtracePath(tree, best_goal_idx);
  std::vector<geometry_msgs::msg::PoseStamped> tree_path = raw_path;
  if (enable_path_shortcut_) {
    tree_path = shortcutPath(raw_path);
  }

  global_path.poses = densifyPath(tree_path);
  const auto now_stamp = node_->now();
  for (auto & pose : global_path.poses) {
    pose.header.frame_id = global_frame_;
    pose.header.stamp = now_stamp;
  }

  RCLCPP_INFO(node_->get_logger(), "规划成功 | Informed RRT* 路径点数: %zu", global_path.poses.size());
  return global_path;
}

}  // namespace nav2_custom_planner

PLUGINLIB_EXPORT_CLASS(nav2_custom_planner::InformedRRTStar, nav2_core::GlobalPlanner)

