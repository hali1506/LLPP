//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"

#include <omp.h>

#include <algorithm>
#include <barrier>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>

#include "ped_waypoint.h"

#ifdef __x86_64__
#include <emmintrin.h>
#include <immintrin.h>
#else
#error "Unknown platform!"
#endif

#ifndef NOCUDA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>

#define SIM_WIDTH 160
#define SIM_HEIGHT 120
#define BORDER_WIDTH 2

typedef struct tile {
  alignas(64);
  int x0, x1;
  int y0, y1;
} tile_t;

typedef struct part {
  alignas(64);
  int tiles_x;
  int tiles_y;
} part_t;

typedef struct threadTask {
  alignas(64);
  Ped::Tagents* agents;
  int start;
  int n_agents;
} task_t;

// Borrowed from GeeksForGeeks
// https://www.geeksforgeeks.org/cpp/thread-pool-in-cpp/

// Class that represents a simple thread pool
class ThreadPool {
 public:
  // // Constructor to creates a thread pool with given
  // number of threads
  ThreadPool(size_t num_threads = std::thread::hardware_concurrency() - 1) {
    // Creating worker threads
    for (size_t i = 0; i < num_threads; ++i) {
      threads_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          // The reason for putting the below code
          // here is to unlock the queue before
          // executing the task so that other
          // threads can perform enqueue tasks
          {
            // Locking the queue so that data
            // can be shared safely
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // Waiting until there is a task to
            // execute or the pool is stopped
            cv_.wait(lock, [this] { return !tasks_.empty() || stop_; });

            // exit the thread in case the pool
            // is stopped and there are no tasks
            if (stop_ && tasks_.empty()) {
              return;
            }

            // Get the next task from the queue
            task = move(tasks_.front());
            tasks_.pop();
          }

          task();
        }
      });
    }
  }

  // Destructor to stop the thread pool
  ~ThreadPool() {
    {
      // Lock the queue to update the stop flag safely
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }

    // Notify all threads
    cv_.notify_all();

    // Joining all worker threads to ensure they have
    // completed their tasks
    for (auto& thread : threads_) {
      thread.join();
    }
  }

  // Enqueue task for execution by the thread pool
  void enqueue(std::function<void()> task) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      tasks_.emplace(move(task));
    }
    cv_.notify_one();
  }

 private:
  // Vector to store worker threads
  std::vector<std::thread> threads_;

  // Queue of tasks
  std::queue<std::function<void()>> tasks_;

  // Mutex to synchronize access to shared data
  std::mutex queue_mutex_;

  // Condition variable to signal changes in the state of
  // the tasks queue
  std::condition_variable cv_;

  // Flag to indicate whether the thread pool should stop
  // or not
  bool stop_ = false;
};

static ThreadPool* pool = nullptr;
static std::barrier<>* barr = nullptr;

static void tickThread(task_t task) {
  for (int i = 0; i < task.n_agents; i++) {
    int pos = task.start + i;
    task.agents->computeNextDesiredPosition(pos);
    task.agents->setX(pos, task.agents->getDesiredX(pos));
    task.agents->setY(pos, task.agents->getDesiredY(pos));
  }

  barr->arrive_and_wait();
}

static part_t compPart(int threads, int gridWidth, int gridHeight) {
  if (threads <= 1) {
    return {1, 1};
  }

  double gridRatio = static_cast<double>(gridWidth) / gridHeight;

  int xBest = 1;
  int yBest = threads;
  int prodBest = std::numeric_limits<int>::max();
  double absBest = std::numeric_limits<double>::max();

  int limit = static_cast<int>(std::sqrt(threads)) + 1;

  for (int x = 1; x <= limit; x++) {
    int y = (threads + x - 1) / x;  // ceil
    int product = x * y;

    double tileRatio = static_cast<double>(x) / y;
    double abs = std::abs(tileRatio - gridRatio);

    if (product < prodBest || (product == prodBest && abs < absBest)) {
      xBest = x;
      yBest = y;
      prodBest = product;
      absBest = abs;
    }
  }

  return {xBest, yBest};
}

static tile_t compute_tile_bounds(int xTile, int yTile, int xTiles, int yTiles,
                                  int width, int height) {
  int x0 = (width * xTile) / xTiles;
  int x1 = (width * (xTile + 1)) / xTiles;

  int y0 = (height * yTile) / yTiles;
  int y1 = (height * (yTile + 1)) / yTiles;

  return {x0, x1, y0, y1};
}

void Ped::Model::setup(std::vector<Ped::AgentsData> agentsInScenario,
                       std::vector<Twaypoint*> destinationsInScenario,
                       IMPLEMENTATION implementation,
                       IMPLEMENTATION_MOVE moveImplementation) {
#ifndef NOCUDA
  // Convenience test: does CUDA work on this machine?
  cuda_test();
#else
  std::cout << "Not compiled for CUDA" << std::endl;
#endif

  // Set up destinations
  destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(),
                                              destinationsInScenario.end());

  // Sets the chosen implementation. Standard in the given code is SEQ
  this->implementation = implementation;

  // Sets whether to use move() or not. Standard is NOMOVE
  this->moveImplementation = moveImplementation;

  tagents = new Tagents(agentsInScenario);
  tagents->setImplementation(implementation);

#ifndef NOCUDA
  if (implementation == CUDA) {
    cudaPed = new CUDAPed(tagents);
  }
#endif

  switch (implementation) {
    case SEQ:
      activeThreads = 1;
      break;
    case OMP:
      activeThreads = omp_get_max_threads();
      break;
    case CPPTHREADS:
      activeThreads = std::thread::hardware_concurrency();
      pool = new ThreadPool(activeThreads - 1);
      barr = new std::barrier<>(activeThreads);
      break;
    case CUDA:
      activeThreads = 1;
      break;
    default:
      activeThreads = 1;
      break;
  }

  if (moveImplementation == MOVE) {
    part_t partition = compPart(activeThreads, SIM_WIDTH, SIM_HEIGHT);
    tilesX = partition.tiles_x;
    tilesY = partition.tiles_y;

    if (tilesX > 1) {
      numVBorders = (tilesX - 1) * tilesY;
    } else {
      numVBorders = 0;
    }

    if (tilesY > 1) {
      numHBorders = (tilesY - 1) * tilesX;
    } else {
      numHBorders = 0;
    }

    numBorderMutexes = numVBorders + numHBorders;
    if (numBorderMutexes > 0) {
      borderLocks = new omp_lock_t[numBorderMutexes];
      for (int i = 0; i < numBorderMutexes; i++) {
        omp_init_lock(&borderLocks[i]);
      }
    }
  }

  // Set up heatmap (relevant for Assignment 4)
  setupHeatmapSeq();
}

void Ped::Model::tick() {
  // EDIT HERE FOR ASSIGNMENT 1
  int nr_agents = tagents->getNumAgents();

  switch (implementation) {
    case SEQ:
      seqTick(nr_agents);
      break;
    case OMP:
      ompTick(nr_agents);
      break;
    case CPPTHREADS:
      cppthreadsTick(nr_agents);
      break;
    case VECTOR:
      simdTick();
      break;
#ifndef NOCUDA
    case CUDA:
      cudaTick();
      break;
#endif
    default:
      seqTick(nr_agents);
      break;
  }
}

#ifndef NOCUDA
void Ped::Model::cudaTick() {
  switch (moveImplementation) {
    case MOVE:
      std::cout << "Move not supported" << std::endl;
      break;
    case NOMOVE:
      cudaPed->CUDA_computeNextDesiredPosition(tagents);
      break;
    default:
      std::cout << "Unknown move, running without move" << std::endl;
      cudaPed->CUDA_computeNextDesiredPosition(tagents);
      break;
  }
}
#endif

void Ped::Model::simdTick() {
  switch (moveImplementation) {
    case MOVE:
      std::cout << "Move not supported" << std::endl;
      break;
    case NOMOVE:
      tagents->computeNextDesiredPositionsSIMD();
      tagents->setXYSIMD();
      break;
    default:
      std::cout << "Unknown move, running without move" << std::endl;
      tagents->computeNextDesiredPositionsSIMD();
      tagents->setXYSIMD();
      break;
  }
}

void Ped::Model::cppthreadsTick(int nr_agents) {
  switch (moveImplementation) {
    case MOVE:
      std::cout << "Move not supported" << std::endl;
      break;
    case NOMOVE:
      // unsigned int n_threads = std::thread::hardware_concurrency();
      // std::thread threads[n_threads - 1];

      // unsigned int agents_per_thread = nr_agents / n_threads;
      // unsigned int agents_rest = nr_agents % n_threads;

      // int start = 0;

      // for (unsigned int i = 0; i < n_threads; i++) {
      //   int n_agents = agents_per_thread;
      //   if (i < agents_rest) {
      //     n_agents++;
      //   }

      //   threadTask task;
      //   task.agents = tagents;
      //   task.n_agents = n_agents;
      //   task.start = start;

      //   start += n_agents;

      //   if (i == n_threads - 1) {
      //     tickThread(task);
      //   } else {
      //     threads[i] = std::thread(tickThread, task);
      //   }
      // }

      // for (int i = 0; i < n_threads - 1; i++) {
      //   threads[i].join();
      // }
      {
        unsigned int agents_per_thread = nr_agents / activeThreads;
        unsigned int agents_rest = nr_agents % activeThreads;

        int start = 0;

        for (unsigned int i = 0; i < activeThreads; i++) {
          int n_agents = agents_per_thread;
          if (i < agents_rest) {
            n_agents++;
          }

          task_t task;
          task.agents = tagents;
          task.n_agents = n_agents;
          task.start = start;

          start += n_agents;

          if (i == activeThreads - 1) {
            tickThread(task);
          } else {
            pool->enqueue([task] { tickThread(task); });
          }
        }
      }
      break;
    default:
      std::cout << "Unknown move, running without move" << std::endl;
      // unsigned int n_threads = std::thread::hardware_concurrency();
      // std::thread threads[n_threads - 1];

      // unsigned int agents_per_thread = nr_agents / n_threads;
      // unsigned int agents_rest = nr_agents % n_threads;

      // int start = 0;

      // for (unsigned int i = 0; i < n_threads; i++) {
      //   int n_agents = agents_per_thread;
      //   if (i < agents_rest) {
      //     n_agents++;
      //   }

      //   threadTask task;
      //   task.agents = tagents;
      //   task.n_agents = n_agents;
      //   task.start = start;

      //   start += n_agents;

      //   if (i == n_threads - 1) {
      //     tickThread(task);
      //   } else {
      //     threads[i] = std::thread(tickThread, task);
      //   }
      // }

      // for (int i = 0; i < n_threads - 1; i++) {
      //   threads[i].join();
      // }
      {
        unsigned int agents_per_thread = nr_agents / activeThreads;
        unsigned int agents_rest = nr_agents % activeThreads;

        int start = 0;

        for (unsigned int i = 0; i < activeThreads; i++) {
          int n_agents = agents_per_thread;
          if (i < agents_rest) {
            n_agents++;
          }

          task_t task;
          task.agents = tagents;
          task.n_agents = n_agents;
          task.start = start;

          start += n_agents;

          if (i == activeThreads - 1) {
            tickThread(task);
          } else {
            pool->enqueue([task] { tickThread(task); });
          }
        }
      }
      break;
  }
}

void Ped::Model::ompTick(int nr_agents) {
  switch (moveImplementation) {
    case MOVE:
#pragma omp parallel for default(none) firstprivate(nr_agents) shared(tagents)
      for (int i = 0; i < nr_agents; i++) {
        tagents->computeNextDesiredPosition(i);
      }
      moveAllOMP();  // Has its own #pragma omp inside
      break;
    case NOMOVE:
#pragma omp parallel for default(none) firstprivate(nr_agents) shared(tagents)
      for (int i = 0; i < nr_agents; i++) {
        tagents->computeNextDesiredPosition(i);
        tagents->setX(i, tagents->getDesiredX(i));
        tagents->setY(i, tagents->getDesiredY(i));
      }
      break;
    default:
      std::cout << "Unknown move, running without move" << std::endl;
#pragma omp parallel for default(none) firstprivate(nr_agents) shared(tagents)
      for (int i = 0; i < nr_agents; i++) {
        tagents->computeNextDesiredPosition(i);
        tagents->setX(i, tagents->getDesiredX(i));
        tagents->setY(i, tagents->getDesiredY(i));
      }
      break;
  }
}

void Ped::Model::seqTick(int nr_agents) {
  switch (moveImplementation) {
    case MOVE:
      for (int i = 0; i < nr_agents; i++) {
        tagents->computeNextDesiredPosition(i);
        move(i);
      }
      break;
    case NOMOVE:
      for (int i = 0; i < nr_agents; i++) {
        tagents->computeNextDesiredPosition(i);
        tagents->setX(i, tagents->getDesiredX(i));
        tagents->setY(i, tagents->getDesiredY(i));
      }
      break;
    default:
      std::cout << "Unknown move, running without move" << std::endl;
      for (int i = 0; i < nr_agents; i++) {
        tagents->computeNextDesiredPosition(i);
        tagents->setX(i, tagents->getDesiredX(i));
        tagents->setY(i, tagents->getDesiredY(i));
      }
      break;
  }
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(int agentId) {
  // Search for neighboring agents
  set<int> neighbors =
      getNeighbors(tagents->getX(agentId), tagents->getY(agentId), 2);

  // Retrieve their positions
  std::vector<std::pair<int, int>> takenPositions;

  // TODO: Double-check if still valid after change
  // for (int neighborIt : neighbors) {
  //   std::pair<int, int> position(tagents->getX(neighborIt),
  //                                tagents->getY(neighborIt));

  for (std::set<int>::iterator neighborIt = neighbors.begin();
       neighborIt != neighbors.end(); ++neighborIt) {
    std::pair<int, int> position(tagents->getX(*neighborIt),
                                 tagents->getY(*neighborIt));
    takenPositions.push_back(position);
  }

  // Compute the three alternative positions that would bring the agent
  // closer to his desiredPosition, starting with the desiredPosition itself
  std::vector<std::pair<int, int>> prioritizedAlternatives;
  std::pair<int, int> pDesired(tagents->getDesiredX(agentId),
                               tagents->getDesiredY(agentId));
  prioritizedAlternatives.push_back(pDesired);

  int diffX = pDesired.first - tagents->getX(agentId);
  int diffY = pDesired.second - tagents->getY(agentId);
  std::pair<int, int> p1, p2;
  if (diffX == 0 || diffY == 0) {
    // Agent wants to walk straight to North, South, West or East
    p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
    p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
  } else {
    // Agent wants to walk diagonally
    p1 = std::make_pair(pDesired.first, tagents->getY(agentId));
    p2 = std::make_pair(tagents->getX(agentId), pDesired.second);
  }
  prioritizedAlternatives.push_back(p1);
  prioritizedAlternatives.push_back(p2);

  // Find the first empty alternative position
  for (std::vector<pair<int, int>>::iterator it =
           prioritizedAlternatives.begin();
       it != prioritizedAlternatives.end(); ++it) {
    // If the current position is not yet taken by any neighbor
    if (std::find(takenPositions.begin(), takenPositions.end(), *it) ==
        takenPositions.end()) {
      // Set the agent's position
      tagents->setX(agentId, (*it).first);
      tagents->setY(agentId, (*it).second);

      break;
    }
  }
}

void Ped::Model::processTile(int x, int y, int tx, int ty, int width,
                             int height, int borderWidth, int numVBorders,
                             std::vector<std::vector<int>>& tileAgents,
                             omp_lock_t* borderLocks, Ped::Tagents* tagents) {
  tile_t bounds = compute_tile_bounds(x, y, tx, ty, width, height);
  int tIndex = y * tx + x;

  int vBorder = tx - 1;
  int hOffset = numVBorders;

  for (int agentId : tileAgents[tIndex]) {
    int agentX = tagents->getX(agentId);
    int agentY = tagents->getY(agentId);

    bool isRight = (x < tx - 1) && (agentX >= bounds.x1 - borderWidth);
    bool isLeft = (x > 0) && (agentX < bounds.x0 + borderWidth);
    bool isBottom = (y < ty - 1) && (agentY >= bounds.y1 - borderWidth);
    bool isTop = (y > 0) && (agentY < bounds.y0 + borderWidth);

    if (!(isRight || isLeft || isBottom || isTop)) {
      move(agentId);
      continue;
    }

    std::vector<omp_lock_t*> locks;
    locks.reserve(4);

    // Vertical borders
    if (isRight) {
      locks.push_back(&borderLocks[y * vBorder + x]);
    }
    if (isLeft) {
      locks.push_back(&borderLocks[y * vBorder + (x - 1)]);
    }

    // Horizontal borders
    if (isBottom) {
      locks.push_back(&borderLocks[hOffset + y * tx + x]);
    }
    if (isTop) {
      locks.push_back(&borderLocks[hOffset + (y - 1) * tx + x]);
    }

    for (omp_lock_t* lock : locks) {
      omp_set_lock(lock);
    }

    move(agentId);

    for (auto it = locks.rbegin(); it != locks.rend(); ++it) {
      omp_unset_lock(*it);
    }
  }
}

void Ped::Model::moveAllOMP() {
  int nr_agents = tagents->getNumAgents();
  int totTiles = tilesX * tilesY;

  std::vector<std::vector<int>> tileAgents(totTiles);
  for (int i = 0; i < nr_agents; i++) {
    int ax = tagents->getX(i);
    int ay = tagents->getY(i);

    int tx = (ax * tilesX) / SIM_WIDTH;
    int ty = (ay * tilesY) / SIM_HEIGHT;
    tx = std::clamp(tx, 0, tilesX - 1);
    ty = std::clamp(ty, 0, tilesY - 1);

    tileAgents[ty * tilesX + tx].push_back(i);
  }

  // Big pragma
#pragma omp parallel for if (implementation == OMP) default(none) \
    firstprivate(tilesX, tilesY, numVBorders)                     \
    shared(tileAgents, borderLocks, tagents) collapse(2)          \
    num_threads(activeThreads) schedule(dynamic)
  for (int tx = 0; tx < tilesX; tx++) {
    for (int ty = 0; ty < tilesY; ty++) {
      processTile(tx, ty, tilesX, tilesY, SIM_WIDTH, SIM_HEIGHT, BORDER_WIDTH,
                  numVBorders, tileAgents, borderLocks, tagents);
    }
  }
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents
/// (search field is a square in the current implementation)
set<int> Ped::Model::getNeighbors(int x, int y, int dist) const {
  // create the output list
  // ( It would be better to include only the agents close by, but this
  // programmer is lazy.)

  // TODO: Uncertain of this change, might not function as intended
  // return set<const Ped::Tagent *>(agents.begin(), agents.end());
  set<int> result;
  for (size_t i = 0; i < tagents->getNumAgents(); i++) {
    result.insert(i);
  }
  return result;
}

void Ped::Model::cleanup() {
  // Nothing to do here right now.
}

Ped::Model::~Model() {
  delete tagents;
  if (borderLocks) {
    for (int i = 0; i < numBorderMutexes; i++) {
      omp_destroy_lock(&borderLocks[i]);
    }
    delete[] borderLocks;
  }
#ifndef NOCUDA
  if (implementation == CUDA) {
    delete cudaPed;
  }
#endif
  std::for_each(destinations.begin(), destinations.end(),
                [](Ped::Twaypoint* destination) { delete destination; });
}
