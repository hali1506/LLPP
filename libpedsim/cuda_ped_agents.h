#pragma once

namespace Ped {
class Tagents;
}

class CUDAPed {
 public:
  typedef struct massData {
    int* x;
    int* y;

    int* desiredPositionX;
    int* desiredPositionY;

    int* waypointX;
    int* waypointY;
    int* waypointR;

    int* lastWaypointX;
    int* lastWaypointY;
    int* lastWaypointR;

    int** agentQueX;
    int** agentQueY;
    int** agentQueR;

    int* agentQueIndex;
    int* agentQueSize;
  } massData_t;

  CUDAPed(Ped::Tagents* agents);
  ~CUDAPed();

  void CUDA_computeNextDesiredPosition(Ped::Tagents* agent);

  void CUDA_cpyToGPUXY(Ped::Tagents* agent);
  void CUDA_cpyToHostXY(Ped::Tagents* agent);

 private:
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

  int** agentQueX_helper;
  int** agentQueY_helper;
  int** agentQueR_helper;

  int* agentQueIndex;
  int* agentQueSize;

  int nrAgents;
};
