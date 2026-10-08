#include "cuda_ped_agents.h"
#include "ped_agents.h"

CUDAPed::CUDAPed(Ped::Tagents* agents) {
  nrAgents = agents->nrAgents;
  cudaMalloc(&x, agents->getXSize());
  cudaMalloc(&y, agents->getYSize());
  cudaMalloc(&desiredPositionX, agents->getDesiredXSize());
  cudaMalloc(&desiredPositionY, agents->getDesiredYSize());
  cudaMalloc(&waypointX, agents->getWaypointXSize());
  cudaMalloc(&waypointY, agents->getWaypointYSize());
  cudaMalloc(&waypointR, agents->getWaypointRSize());
  cudaMalloc(&lastWaypointX, agents->getLastWaypointXSize());
  cudaMalloc(&lastWaypointY, agents->getLastWaypointYSize());
  cudaMalloc(&lastWaypointR, agents->getLastWaypointRSize());

  agentQueX_helper = (int**)malloc(nrAgents * sizeof(int*));
  agentQueY_helper = (int**)malloc(nrAgents * sizeof(int*));
  agentQueR_helper = (int**)malloc(nrAgents * sizeof(int*));

  for (size_t i = 0; i < agents->nrAgents; i++) {
    cudaMalloc(&agentQueX_helper[i], agents->getAgentQueXInnerSize(i));
    cudaMalloc(&agentQueY_helper[i], agents->getAgentQueYInnerSize(i));
    cudaMalloc(&agentQueR_helper[i], agents->getAgentQueRInnerSize(i));
  }

  cudaMalloc(&agentQueX, agents->getAgentQueXSize());
  cudaMalloc(&agentQueY, agents->getAgentQueYSize());
  cudaMalloc(&agentQueR, agents->getAgentQueRSize());
  cudaMalloc(&agentQueIndex, agents->getAgentQueIndexSize());
  cudaMalloc(&agentQueSize, agents->getAgentQueSizeSize());

  // Copy all the memory from the host to the GPU
  // Do not copy over X and Y since written to before call
  cudaMemcpy(desiredPositionX, agents->desiredPositionX,
             agents->getDesiredXSize(), cudaMemcpyHostToDevice);
  cudaMemcpy(desiredPositionY, agents->desiredPositionY,
             agents->getDesiredYSize(), cudaMemcpyHostToDevice);
  cudaMemcpy(waypointX, agents->waypointX, agents->getWaypointXSize(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(waypointY, agents->waypointY, agents->getWaypointYSize(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(waypointR, agents->waypointR, agents->getWaypointRSize(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(lastWaypointX, agents->lastWaypointX,
             agents->getLastWaypointXSize(), cudaMemcpyHostToDevice);
  cudaMemcpy(lastWaypointY, agents->lastWaypointY,
             agents->getLastWaypointYSize(), cudaMemcpyHostToDevice);
  cudaMemcpy(lastWaypointR, agents->lastWaypointR,
             agents->getLastWaypointRSize(), cudaMemcpyHostToDevice);

  for (size_t i = 0; i < nrAgents; i++) {
    cudaMemcpy(agentQueX_helper[i], agents->agentQueX[i],
               agents->getAgentQueXInnerSize(i), cudaMemcpyHostToDevice);
    cudaMemcpy(agentQueY_helper[i], agents->agentQueY[i],
               agents->getAgentQueYInnerSize(i), cudaMemcpyHostToDevice);
    cudaMemcpy(agentQueR_helper[i], agents->agentQueR[i],
               agents->getAgentQueRInnerSize(i), cudaMemcpyHostToDevice);
  }

  cudaMemcpy(agentQueX, agentQueX_helper, agents->getAgentQueXSize(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(agentQueY, agentQueY_helper, agents->getAgentQueYSize(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(agentQueR, agentQueR_helper, agents->getAgentQueRSize(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(agentQueIndex, agents->agentQueIndex,
             agents->getAgentQueIndexSize(), cudaMemcpyHostToDevice);
  cudaMemcpy(agentQueSize, agents->agentQueSize, agents->getAgentQueSizeSize(),
             cudaMemcpyHostToDevice);
}

CUDAPed::~CUDAPed() {
  cudaFree(x);
  cudaFree(y);
  cudaFree(desiredPositionX);
  cudaFree(desiredPositionY);
  cudaFree(waypointX);
  cudaFree(waypointY);
  cudaFree(waypointR);
  cudaFree(lastWaypointX);
  cudaFree(lastWaypointY);
  cudaFree(lastWaypointR);

  for (size_t i = 0; i < nrAgents; i++) {
    cudaFree(agentQueX_helper);
    cudaFree(agentQueY_helper);
    cudaFree(agentQueR_helper);
  }

  cudaFree(agentQueX);
  cudaFree(agentQueY);
  cudaFree(agentQueR);
  cudaFree(agentQueIndex);
  cudaFree(agentQueSize);
  free(agentQueX_helper);
  free(agentQueY_helper);
  free(agentQueR_helper);
}

__global__ void computeNextDesiredPositionHelper(CUDAPed::massData_t* data,
                                                 int nrAgents) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nrAgents) return;

  bool isNull = (data->waypointX[i] == 0 && data->waypointY[i] == 0 &&
                 data->waypointR[i] == 0);
  bool reached = false;
  if (!isNull) {
    float diffX = data->waypointX[i] - data->x[i];
    float diffY = data->waypointY[i] - data->y[i];
    float length = sqrt(diffX * diffX + diffY * diffY);
    reached = length < data->waypointR[i];
  }

  if ((reached || isNull) && data->agentQueSize[i] > 0) {
    int index = data->agentQueIndex[i];
    data->waypointX[i] = data->agentQueX[i][index];
    data->waypointY[i] = data->agentQueY[i][index];
    data->waypointR[i] = data->agentQueR[i][index];
    data->agentQueIndex[i] = (index + 1) % data->agentQueSize[i];
  }

  if (data->waypointX[i] == 0 && data->waypointY[i] == 0 &&
      data->waypointR[i] == 0) {
    // no destination, no need to compute where to move to
    return;
  }

  double diffX = data->waypointX[i] - data->x[i];
  double diffY = data->waypointY[i] - data->y[i];
  double len = sqrt(diffX * diffX + diffY * diffY);
  data->desiredPositionX[i] = (int)round(data->x[i] + diffX / len);
  data->desiredPositionY[i] = (int)round(data->y[i] + diffY / len);
}

void CUDAPed::CUDA_computeNextDesiredPosition(Ped::Tagents* agent) {
  CUDA_cpyToGPUXY(agent);

  massData_t bigData;
  bigData.x = x;
  bigData.y = y;
  bigData.desiredPositionX = desiredPositionX;
  bigData.desiredPositionY = desiredPositionY;
  bigData.waypointX = waypointX;
  bigData.waypointY = waypointY;
  bigData.waypointR = waypointR;
  bigData.lastWaypointX = lastWaypointX;
  bigData.lastWaypointY = lastWaypointY;
  bigData.lastWaypointR = lastWaypointR;
  bigData.agentQueX = agentQueX;
  bigData.agentQueY = agentQueY;
  bigData.agentQueR = agentQueR;
  bigData.agentQueIndex = agentQueIndex;
  bigData.agentQueSize = agentQueSize;

  massData_t* d_bigData;
  cudaMalloc(&d_bigData, sizeof(massData_t));
  cudaMemcpy(d_bigData, &bigData, sizeof(massData_t), cudaMemcpyHostToDevice);

  computeNextDesiredPositionHelper<<<(agent->nrAgents + 255) / 256, 256>>>(
      d_bigData, agent->nrAgents);
  cudaDeviceSynchronize();

  cudaFree(d_bigData);

  CUDA_cpyToHostXY(agent);
}

void CUDAPed::CUDA_cpyToGPUXY(Ped::Tagents* agent) {
  cudaMemcpy(x, agent->x, agent->getXSize(), cudaMemcpyHostToDevice);
  cudaMemcpy(y, agent->y, agent->getYSize(), cudaMemcpyHostToDevice);
}

void CUDAPed::CUDA_cpyToHostXY(Ped::Tagents* agent) {
  cudaMemcpy(agent->x, desiredPositionX,
             agent->getDesiredXSize(), cudaMemcpyDeviceToHost);
  cudaMemcpy(agent->y, desiredPositionY,
             agent->getDesiredYSize(), cudaMemcpyDeviceToHost);
}