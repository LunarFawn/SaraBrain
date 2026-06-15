#include <vector>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <cstring>
#include <string>

// Non-propagating relations like "is_a" are handled at the ingestion point
// into the C++ engine to keep the hot path tight.

struct Edge {
    int node_id; // The "other" side of the edge
    float strength;
};

struct ResultNode {
    int id;
    float weight;
};

class SaraEngine {
public:
    // Adjacency lists
    std::unordered_map<int, std::vector<Edge>> outgoing;
    std::unordered_map<int, std::vector<Edge>> incoming;

    void add_segment(int src, int tgt, float strength, bool propagating) {
        if (!propagating) return;
        outgoing[src].push_back({tgt, strength});
        incoming[tgt].push_back({src, strength});
    }

    void clear() {
        outgoing.clear();
        incoming.clear();
    }

    // BFS that computes max average path weight for each reached node.
    // This matches the logic of Recognizer._path_weight and propagate_into.
    int propagate(int start_node, int max_depth, float min_strength, 
                  bool bidirectional, ResultNode* out_results, int max_results) {
        
        // reached_id -> best_average_weight
        std::unordered_map<int, float> reached;
        
        struct State {
            int node;
            int depth;
            float total_strength;
            int length;
        };

        std::queue<State> q;
        q.push({start_node, 0, 0.0f, 0});
        // We don't add the start node to 'reached' because propagate_into 
        // explicitly skips it: if target_id == seed.id: continue
        
        // To avoid cycles and redundant work, we track the best weight seen.
        // In a true BFS, the first time we see a node it's at the shortest depth,
        // but not necessarily the highest weight.
        
        while (!q.empty()) {
            State curr = q.front();
            q.pop();

            if (curr.depth >= max_depth) continue;

            // lambda to process a list of edges
            auto process = [&](const std::vector<Edge>& edges) {
                for (const auto& edge : edges) {
                    if (edge.strength < min_strength) continue;
                    
                    float new_total = curr.total_strength + edge.strength;
                    int new_len = curr.length + 1;
                    float avg = new_total / new_len;

                    if (reached.find(edge.node_id) == reached.end() || avg > reached[edge.node_id]) {
                        reached[edge.node_id] = avg;
                        q.push({edge.node_id, curr.depth + 1, new_total, new_len});
                    }
                }
            };

            if (outgoing.count(curr.node)) {
                process(outgoing[curr.node]);
            }
            
            if (bidirectional && incoming.count(curr.node)) {
                process(incoming[curr.node]);
            }
        }

        int count = 0;
        for (auto const& [id, weight] : reached) {
            if (count >= max_results) break;
            out_results[count++] = {id, weight};
        }
        return count;
    }
};

// C-compatible API for ctypes
extern "C" {
    SaraEngine* engine_create() {
        return new SaraEngine();
    }

    void engine_destroy(SaraEngine* e) {
        delete e;
    }

    void engine_add_segment(SaraEngine* e, int src, int tgt, float strength, bool propagating) {
        e->add_segment(src, tgt, strength, propagating);
    }

    void engine_clear(SaraEngine* e) {
        e->clear();
    }

    int engine_propagate(SaraEngine* e, int start_node, int max_depth, float min_strength, 
                         bool bidirectional, ResultNode* out_results, int max_results) {
        return e->propagate(start_node, max_depth, min_strength, bidirectional, out_results, max_results);
    }
}
