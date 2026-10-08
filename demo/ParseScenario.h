//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2015
//

#ifndef _parsescenario_h_
#define _parsescenario_h_

#include <tinyxml2.h>

#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ped_agents.h"
#include "ped_waypoint.h"

using namespace std;
using namespace tinyxml2;

class ParseScenario {
 public:
  ParseScenario() {}
  ParseScenario(std::string filename, bool verbose = false);
  ~ParseScenario() {}

  // returns the collection of agents defined by this scenario
  vector<Ped::AgentsData> getAgents() const;

  // contains all defined waypoints
  vector<Ped::Twaypoint*> getWaypoints();

 private:
  XMLDocument doc;

  // final collection of all created agents
  vector<Ped::AgentsData> agents;

  // temporary collection of agents used to
  // keep track of all agents that are generated
  // within the current opened agents xml tag
  vector<Ped::AgentsData> tempAgents;

  // contains all defined waypoints
  map<string, Ped::Twaypoint*> waypoints;
};

#endif
