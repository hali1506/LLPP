#include "ped_agents.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "ped_waypoint.h"

#ifdef __x86_64__
#include <emmintrin.h>
#include <immintrin.h>
#else
#error "Unknown platform!"
#endif

#define ALIGNMENT 256

Ped::Tagents::Tagents(std::vector<AgentsData>& agentsData) {
  size_t n = agentsData.size();
  this->x = static_cast<int*>(_mm_malloc(n * sizeof(int), ALIGNMENT));
  this->y = static_cast<int*>(_mm_malloc(n * sizeof(int), ALIGNMENT));
  this->desiredPositionX =
      static_cast<int*>(_mm_malloc(n * sizeof(int), ALIGNMENT));
  this->desiredPositionY =
      static_cast<int*>(_mm_malloc(n * sizeof(int), ALIGNMENT));
  this->waypointX = static_cast<int*>(_mm_malloc(n * sizeof(int), ALIGNMENT));
  this->waypointY = static_cast<int*>(_mm_malloc(n * sizeof(int), ALIGNMENT));
  this->waypointR = static_cast<int*>(_mm_malloc(n * sizeof(int), ALIGNMENT));
  this->lastWaypointX =
      static_cast<int*>(_mm_malloc(n * sizeof(int), ALIGNMENT));
  this->lastWaypointY =
      static_cast<int*>(_mm_malloc(n * sizeof(int), ALIGNMENT));
  this->lastWaypointR =
      static_cast<int*>(_mm_malloc(n * sizeof(int), ALIGNMENT));
  this->agentQueX = static_cast<int**>(_mm_malloc(n * sizeof(int*), ALIGNMENT));
  this->agentQueY = static_cast<int**>(_mm_malloc(n * sizeof(int*), ALIGNMENT));
  this->agentQueR = static_cast<int**>(_mm_malloc(n * sizeof(int*), ALIGNMENT));
  this->agentQueIndex =
      static_cast<int*>(_mm_malloc(n * sizeof(int), ALIGNMENT));
  this->agentQueSize =
      static_cast<int*>(_mm_malloc(n * sizeof(int), ALIGNMENT));

  // Zero-initialize all allocations
  std::memset(this->x, 0, n * sizeof(int));
  std::memset(this->y, 0, n * sizeof(int));
  std::memset(this->desiredPositionX, 0, n * sizeof(int));
  std::memset(this->desiredPositionY, 0, n * sizeof(int));
  std::memset(this->waypointX, 0, n * sizeof(int));
  std::memset(this->waypointY, 0, n * sizeof(int));
  std::memset(this->waypointR, 0, n * sizeof(int));
  std::memset(this->lastWaypointX, 0, n * sizeof(int));
  std::memset(this->lastWaypointY, 0, n * sizeof(int));
  std::memset(this->lastWaypointR, 0, n * sizeof(int));
  std::memset(this->agentQueX, 0, n * sizeof(int*));
  std::memset(this->agentQueY, 0, n * sizeof(int*));
  std::memset(this->agentQueR, 0, n * sizeof(int*));
  std::memset(this->agentQueIndex, 0, n * sizeof(int));
  std::memset(this->agentQueSize, 0, n * sizeof(int));
  this->nrAgents = n;

  for (size_t i = 0; i < n; i++) {
    this->x[i] = agentsData[i].x;
    this->y[i] = agentsData[i].y;

    size_t waypointCount = agentsData[i].waypoints.size();
    this->agentQueX[i] =
        static_cast<int*>(_mm_malloc(waypointCount * sizeof(int), ALIGNMENT));
    this->agentQueY[i] =
        static_cast<int*>(_mm_malloc(waypointCount * sizeof(int), ALIGNMENT));
    this->agentQueR[i] =
        static_cast<int*>(_mm_malloc(waypointCount * sizeof(int), ALIGNMENT));

    if (waypointCount > 0) {
      std::memset(this->agentQueX[i], 0, waypointCount * sizeof(int));
      std::memset(this->agentQueY[i], 0, waypointCount * sizeof(int));
      std::memset(this->agentQueR[i], 0, waypointCount * sizeof(int));
    }

    size_t index = 0;
    for (auto& wp : agentsData[i].waypoints) {
      this->agentQueX[i][index] = wp->getx();
      this->agentQueY[i][index] = wp->gety();
      this->agentQueR[i][index] = wp->getr();
      index++;
    }
    this->agentQueSize[i] = waypointCount;
  }
}

Ped::Tagents::~Tagents() {
  // Free all aligned allocations
  for (size_t i = 0; i < nrAgents; i++) {
    if (agentQueX[i]) _mm_free(agentQueX[i]);
    if (agentQueY[i]) _mm_free(agentQueY[i]);
    if (agentQueR[i]) _mm_free(agentQueR[i]);
  }

  _mm_free(x);
  _mm_free(y);
  _mm_free(desiredPositionX);
  _mm_free(desiredPositionY);
  _mm_free(waypointX);
  _mm_free(waypointY);
  _mm_free(waypointR);
  _mm_free(lastWaypointX);
  _mm_free(lastWaypointY);
  _mm_free(lastWaypointR);
  _mm_free(agentQueX);
  _mm_free(agentQueY);
  _mm_free(agentQueR);
  _mm_free(agentQueIndex);
  _mm_free(agentQueSize);
}

// Used for all except VECTOR
void Ped::Tagents::updateNextDestination(int i) {
  bool isNull = (waypointX[i] == 0 && waypointY[i] == 0 && waypointR[i] == 0);
  bool reached = false;
  if (!isNull) {
    float diffX = waypointX[i] - x[i];
    float diffY = waypointY[i] - y[i];
    float length = sqrt(diffX * diffX + diffY * diffY);
    reached = length < waypointR[i];
  }

  if ((reached || isNull) && agentQueSize[i] > 0) {
    int index = agentQueIndex[i];
    waypointX[i] = agentQueX[i][index];
    waypointY[i] = agentQueY[i][index];
    waypointR[i] = agentQueR[i][index];
    agentQueIndex[i] = (index + 1) % agentQueSize[i];
  }
}

void Ped::Tagents::computeNextDesiredPosition(int i) {
  updateNextDestination(i);

  if (waypointX[i] == 0 && waypointY[i] == 0 && waypointR[i] == 0) {
    // no destination, no need to compute where to move to
    return;
  }

  double diffX = waypointX[i] - x[i];
  double diffY = waypointY[i] - y[i];
  double len = sqrt(diffX * diffX + diffY * diffY);
  desiredPositionX[i] = (int)round(x[i] + diffX / len);
  desiredPositionY[i] = (int)round(y[i] + diffY / len);
}

// Only used for VECTOR
void Ped::Tagents::updateNextDestinationsSIMD() {
  size_t agentCount = getNumAgents();
  int* shouldCycle =
      static_cast<int*>(_mm_malloc(agentCount * sizeof(int), ALIGNMENT));

  size_t simdCount = agentCount - (agentCount % 8);
  __m256i zero = _mm256_setzero_si256();

  for (size_t i = 0; i < simdCount; i += 8) {
    __m256i wp_x = _mm256_load_si256((__m256i*)&waypointX[i]);
    __m256i wp_y = _mm256_load_si256((__m256i*)&waypointY[i]);
    __m256i wp_r = _mm256_load_si256((__m256i*)&waypointR[i]);

    __m256i x_i = _mm256_load_si256((__m256i*)&x[i]);
    __m256i y_i = _mm256_load_si256((__m256i*)&y[i]);

    __m256i is_null =
        _mm256_and_si256(_mm256_cmpeq_epi32(wp_x, zero),
                         _mm256_and_si256(_mm256_cmpeq_epi32(wp_y, zero),
                                          _mm256_cmpeq_epi32(wp_r, zero)));

    __m256i diff_x = _mm256_sub_epi32(wp_x, x_i);
    __m256i diff_y = _mm256_sub_epi32(wp_y, y_i);
    __m256 dx_f = _mm256_cvtepi32_ps(diff_x);
    __m256 dy_f = _mm256_cvtepi32_ps(diff_y);
    __m256 dist_sq =
        _mm256_add_ps(_mm256_mul_ps(dx_f, dx_f), _mm256_mul_ps(dy_f, dy_f));

    __m256 r_f = _mm256_cvtepi32_ps(wp_r);
    __m256 r_sq = _mm256_mul_ps(r_f, r_f);

    __m256 reached = _mm256_cmp_ps(dist_sq, r_sq, _CMP_LT_OQ);

    __m256i result = _mm256_or_si256(is_null, _mm256_castps_si256(reached));
    result = _mm256_and_si256(result, _mm256_set1_epi32(1));
    _mm256_store_si256((__m256i*)&shouldCycle[i], result);
  }

  // Rest
  size_t i = simdCount;
  if (i < agentCount) {
    __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    int remaining = agentCount - i;
    __m256i rem_vec = _mm256_set1_epi32(remaining);
    __m256i mask = _mm256_cmpgt_epi32(rem_vec, indices);

    __m256i wp_x = _mm256_maskload_epi32((const int*)&waypointX[i], mask);
    __m256i wp_y = _mm256_maskload_epi32((const int*)&waypointY[i], mask);
    __m256i wp_r = _mm256_maskload_epi32((const int*)&waypointR[i], mask);

    __m256i x_i = _mm256_maskload_epi32((const int*)&x[i], mask);
    __m256i y_i = _mm256_maskload_epi32((const int*)&y[i], mask);

    __m256i is_null =
        _mm256_and_si256(_mm256_cmpeq_epi32(wp_x, zero),
                         _mm256_and_si256(_mm256_cmpeq_epi32(wp_y, zero),
                                          _mm256_cmpeq_epi32(wp_r, zero)));

    __m256i diff_x = _mm256_sub_epi32(wp_x, x_i);
    __m256i diff_y = _mm256_sub_epi32(wp_y, y_i);
    __m256 dx_f = _mm256_cvtepi32_ps(diff_x);
    __m256 dy_f = _mm256_cvtepi32_ps(diff_y);
    __m256 dist_sq =
        _mm256_add_ps(_mm256_mul_ps(dx_f, dx_f), _mm256_mul_ps(dy_f, dy_f));

    __m256 r_f = _mm256_cvtepi32_ps(wp_r);
    __m256 r_sq = _mm256_mul_ps(r_f, r_f);

    __m256 reached = _mm256_cmp_ps(dist_sq, r_sq, _CMP_LT_OQ);

    __m256i result = _mm256_or_si256(is_null, _mm256_castps_si256(reached));
    result = _mm256_and_si256(result, _mm256_set1_epi32(1));
    _mm256_maskstore_epi32((int*)&shouldCycle[i], mask, result);
  }

  for (size_t i = 0; i < agentCount; i++) {
    if (shouldCycle[i] && agentQueSize[i] > 0) {
      int index = agentQueIndex[i];
      waypointX[i] = agentQueX[i][index];
      waypointY[i] = agentQueY[i][index];
      waypointR[i] = agentQueR[i][index];
      agentQueIndex[i] = (index + 1) % agentQueSize[i];
    }
  }

  _mm_free(shouldCycle);
}

void Ped::Tagents::computeNextDesiredPositionsSIMD() {
  updateNextDestinationsSIMD();

  size_t length = getNumAgents();
  size_t simdLength = length - (length % 8);

  for (size_t i = 0; i < simdLength; i += 8) {
    __m256i wpx = _mm256_load_si256((__m256i*)&waypointX[i]);
    __m256i wpy = _mm256_load_si256((__m256i*)&waypointY[i]);

    __m256i x_i = _mm256_load_si256((__m256i*)&x[i]);
    __m256i y_i = _mm256_load_si256((__m256i*)&y[i]);

    __m256i diff_x_i = _mm256_sub_epi32(wpx, x_i);
    __m256i diff_y_i = _mm256_sub_epi32(wpy, y_i);

    __m256 diff_x_f = _mm256_cvtepi32_ps(diff_x_i);
    __m256 diff_y_f = _mm256_cvtepi32_ps(diff_y_i);

    __m256 diff_x_sqr = _mm256_mul_ps(diff_x_f, diff_x_f);
    __m256 diff_y_sqr = _mm256_mul_ps(diff_y_f, diff_y_f);

    __m256 len = _mm256_add_ps(diff_x_sqr, diff_y_sqr);
    len = _mm256_sqrt_ps(len);

    __m256 desired = _mm256_div_ps(diff_x_f, len);
    desired = _mm256_add_ps(_mm256_cvtepi32_ps(x_i), desired);
    desired = _mm256_round_ps(desired, _MM_FROUND_TO_NEAREST_INT);
    _mm256_store_si256((__m256i*)&desiredPositionX[i],
                       _mm256_cvtps_epi32(desired));

    desired = _mm256_div_ps(diff_y_f, len);
    desired = _mm256_add_ps(_mm256_cvtepi32_ps(y_i), desired);
    desired = _mm256_round_ps(desired, _MM_FROUND_TO_NEAREST_INT);
    _mm256_store_si256((__m256i*)&desiredPositionY[i],
                       _mm256_cvtps_epi32(desired));
  }

  // Rest
  size_t i = simdLength;
  if (i < length) {
    __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    int remaining = length - i;
    __m256i rem_vec = _mm256_set1_epi32(remaining);
    __m256i mask = _mm256_cmpgt_epi32(rem_vec, indices);

    __m256i wpx = _mm256_maskload_epi32((const int*)&waypointX[i], mask);
    __m256i wpy = _mm256_maskload_epi32((const int*)&waypointY[i], mask);

    __m256i x_i = _mm256_maskload_epi32((const int*)&x[i], mask);
    __m256i y_i = _mm256_maskload_epi32((const int*)&y[i], mask);

    __m256i diff_x_i = _mm256_sub_epi32(wpx, x_i);
    __m256i diff_y_i = _mm256_sub_epi32(wpy, y_i);

    __m256 diff_x_f = _mm256_cvtepi32_ps(diff_x_i);
    __m256 diff_y_f = _mm256_cvtepi32_ps(diff_y_i);

    __m256 diff_x_sqr = _mm256_mul_ps(diff_x_f, diff_x_f);
    __m256 diff_y_sqr = _mm256_mul_ps(diff_y_f, diff_y_f);

    __m256 len = _mm256_add_ps(diff_x_sqr, diff_y_sqr);
    len = _mm256_sqrt_ps(len);

    __m256 desired = _mm256_div_ps(diff_x_f, len);
    desired = _mm256_add_ps(_mm256_cvtepi32_ps(x_i), desired);
    desired = _mm256_round_ps(desired, _MM_FROUND_TO_NEAREST_INT);
    _mm256_maskstore_epi32((int*)&desiredPositionX[i], mask,
                           _mm256_cvtps_epi32(desired));

    desired = _mm256_div_ps(diff_y_f, len);
    desired = _mm256_add_ps(_mm256_cvtepi32_ps(y_i), desired);
    desired = _mm256_round_ps(desired, _MM_FROUND_TO_NEAREST_INT);
    _mm256_maskstore_epi32((int*)&desiredPositionY[i], mask,
                           _mm256_cvtps_epi32(desired));
  }
}

void Ped::Tagents::setXYSIMD() {
  size_t length = getNumAgents();
  size_t simdLength = length - (length % 8);

  for (size_t i = 0; i < simdLength; i += 8) {
    __m256i dx = _mm256_load_si256((__m256i*)&desiredPositionX[i]);
    __m256i dy = _mm256_load_si256((__m256i*)&desiredPositionY[i]);
    _mm256_store_si256((__m256i*)&x[i], dx);
    _mm256_store_si256((__m256i*)&y[i], dy);
  }

  // Rest
  size_t i = simdLength;
  if (i < length) {
    __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    int remaining = length - i;
    __m256i rem_vec = _mm256_set1_epi32(remaining);
    __m256i mask = _mm256_cmpgt_epi32(rem_vec, indices);

    __m256i dx = _mm256_maskload_epi32((const int*)&desiredPositionX[i], mask);
    __m256i dy = _mm256_maskload_epi32((const int*)&desiredPositionY[i], mask);
    _mm256_maskstore_epi32((int*)&x[i], mask, dx);
    _mm256_maskstore_epi32((int*)&y[i], mask, dy);
  }
}
