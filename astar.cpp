#include <vector>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <tuple>

using namespace std;

typedef tuple<int, int> Point;

struct PointHash {
    size_t operator()(const Point& p) const {
        auto [x, y] = p;
        return hash<int>()(x) ^ hash<int>()(y);
    }
};

struct PointEqual {
    bool operator()(const Point& a, const Point& b) const {
        return a == b;
    }
};

int heuristic(const Point& a, const Point& b) {
    auto [ax, ay] = a;
    auto [bx, by] = b;
    return abs(ax - bx) + abs(ay - by);
}

vector<Point> astar(const vector<vector<int>>& grid, Point start, Point goal) {
    priority_queue<pair<int, Point>, vector<pair<int, Point>>, greater<>> open_set;
    unordered_map<Point, Point, PointHash, PointEqual> came_from;
    unordered_map<Point, int, PointHash, PointEqual> g_score, f_score;

    open_set.push({0, start});
    g_score[start] = 0;
    f_score[start] = heuristic(start, goal);

    vector<Point> neighbors = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    while (!open_set.empty()) {
        auto [_, current] = open_set.top();
        open_set.pop();

        if (current == goal) {
            vector<Point> path;
            while (came_from.find(current) != came_from.end()) {
                path.push_back(current);
                current = came_from[current];
            }
            path.push_back(start);
            reverse(path.begin(), path.end());
            return path;
        }

        for (auto& offset : neighbors) {
            Point neighbor = {get<0>(current) + get<0>(offset), get<1>(current) + get<1>(offset)};
            if (0 <= get<0>(neighbor) && get<0>(neighbor) < grid.size() && 0 <= get<1>(neighbor) && get<1>(neighbor) < grid[0].size()) {
                if (grid[get<0>(neighbor)][get<1>(neighbor)] == 2) continue;  // Assuming 2 is an obstacle
                int tentative_g_score = g_score[current] + 1;
                if (!g_score.count(neighbor) || tentative_g_score < g_score[neighbor]) {
                    came_from[neighbor] = current;
                    g_score[neighbor] = tentative_g_score;
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal);
                    open_set.push({f_score[neighbor], neighbor});
                }
            }
        }
    }
    return {};
}
