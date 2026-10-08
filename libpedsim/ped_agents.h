#pragma once
#include <vector>

#ifdef __x86_64__
#include <emmintrin.h>
#include <immintrin.h>
#else
#error "Unknown platform!"
#endif

namespace Ped {

// The implementation modes for Assignment 1 + 2:
// chooses which implementation to use for tick()
enum IMPLEMENTATION { CUDA, VECTOR, OMP, CPPTHREADS, SEQ };
enum IMPLEMENTATION_MOVE { MOVE, NOMOVE };

class Twaypoint;

struct AgentsData {
  int x, y;
  std::vector<Twaypoint*> waypoints;
};

class Tagents {
 public:
  Tagents() {}
  Tagents(std::vector<AgentsData>& agentsData);
  ~Tagents();

  size_t getNumAgents() const { return nrAgents; }

  // Returns the coordinates of the desired position
  int getDesiredX(size_t agentId) const { return desiredPositionX[agentId]; }
  int getDesiredY(size_t agentId) const { return desiredPositionY[agentId]; }

  // TODO: Refactor to use getter for SIMD
#ifdef __AVX2__
  __m256i getDesiredX_vec(int agent_id) const {
    return _mm256_load_si256((__m256i*)&desiredPositionX[agent_id]);
  }
  __m256i getDesiredY_vec(int agent_id) const {
    return _mm256_load_si256((__m256i*)&desiredPositionY[agent_id]);
  }
#endif

  // Sets the agent's position
  void setX(size_t agentId, int newX) { x[agentId] = newX; }
  void setY(size_t agentId, int newY) { y[agentId] = newY; }

  // TODO: Refactor to use setter for SIMD
#ifdef __AVX2__
  void setX_vec(int agent_id, __m256i newX) {
    _mm256_store_si256((__m256i*)&x[agent_id], newX);
  }
  void setY_vec(int agent_id, __m256i newY) {
    _mm256_store_si256((__m256i*)&y[agent_id], newY);
  }
#endif
  // SIMD set all agent's positions, VECTOR
  void setXYSIMD();

  // Single-agent compute desired position, non-VECTOR
  void computeNextDesiredPosition(int i);

  // SIMD compute all desired positions, VECTOR
  void computeNextDesiredPositionsSIMD();

  // Position of agent defined by x and y
  int getX(size_t agentId) const { return x[agentId]; };
  int getY(size_t agentId) const { return y[agentId]; };

  // TODO: Refactor to use getter for SIMD
#ifdef __AVX2__
  __m256i getX_vec(int agent_id) const {
    return _mm256_load_si256((__m256i*)&x[agent_id]);
  }
  __m256i getY_vec(int agent_id) const {
    return _mm256_load_si256((__m256i*)&y[agent_id]);
  }
#endif

  // Position of agent defined by x and y
  int getWaypointX(size_t agentId) const { return waypointX[agentId]; };
  int getWaypointY(size_t agentId) const { return waypointY[agentId]; };
  int getWaypointR(size_t agentId) const { return waypointR[agentId]; };

  // TODO: Refactor to use getter for SIMD
#ifdef __AVX2__
  __m256i getWaypointX_vec(int agent_id) const {
    return _mm256_load_si256((__m256i*)&waypointX[agent_id]);
  }
  __m256i getWaypointY_vec(int agent_id) const {
    return _mm256_load_si256((__m256i*)&waypointY[agent_id]);
  }
  __m256i getWaypointR_vec(int agent_id) const {
    return _mm256_load_si256((__m256i*)&waypointR[agent_id]);
  }
#endif

  // Sizes
  int getDesiredXSize() const { return nrAgents * sizeof(int); }
  int getDesiredYSize() const { return nrAgents * sizeof(int); }

  int getXSize() const { return nrAgents * sizeof(int); }
  int getYSize() const { return nrAgents * sizeof(int); }

  int getWaypointXSize() const { return nrAgents * sizeof(int); }
  int getWaypointYSize() const { return nrAgents * sizeof(int); }
  int getWaypointRSize() const { return nrAgents * sizeof(int); }

  int getLastWaypointXSize() const { return nrAgents * sizeof(int); }
  int getLastWaypointYSize() const { return nrAgents * sizeof(int); }
  int getLastWaypointRSize() const { return nrAgents * sizeof(int); }

  int getAgentQueXSize() const { return nrAgents * sizeof(int*); }
  int getAgentQueYSize() const { return nrAgents * sizeof(int*); }
  int getAgentQueRSize() const { return nrAgents * sizeof(int*); }

  // Inner array sizes per agent (based on waypoint count)
  int getAgentQueXInnerSize(size_t agentId) const {
    return agentQueSize[agentId] * sizeof(int);
  }
  int getAgentQueYInnerSize(size_t agentId) const {
    return agentQueSize[agentId] * sizeof(int);
  }
  int getAgentQueRInnerSize(size_t agentId) const {
    return agentQueSize[agentId] * sizeof(int);
  }

  int getAgentQueIndexSize() const { return nrAgents * sizeof(int); }
  int getAgentQueSizeSize() const { return nrAgents * sizeof(int); }

  // Sets which implementation to use
  void setImplementation(IMPLEMENTATION i) { impl = i; };
  void setImplementationMove(IMPLEMENTATION_MOVE i) { impl_move = i; };

  // The agent's current position
  int* x;
  int* y;

  // The agent's desired next position
  int* desiredPositionX;
  int* desiredPositionY;

  // The current destination (may require several steps to reach)
  int* waypointX;
  int* waypointY;
  int* waypointR;

  // The last destination
  int* lastWaypointX;
  int* lastWaypointY;
  int* lastWaypointR;

  // Ring buffer instead of deque
  int** agentQueX;
  int** agentQueY;
  int** agentQueR;

  int* agentQueIndex;
  int* agentQueSize;
  size_t nrAgents = 0;

 private:
  IMPLEMENTATION impl = SEQ;
  IMPLEMENTATION_MOVE impl_move = NOMOVE;

  // Single-agent destination update
  void updateNextDestination(int i);

  // SIMD destination update
  void updateNextDestinationsSIMD();
};
}  // namespace Ped
