//
// Adapted for Low Level Parallel Programming 2017
//
// Twaypoint represents a destination in a scenario.
// It is described by its position (x,y) and a radius
// which is used to define how close an agent has to be
// to this destination until we consider that the agent
// has reached its destination.int Ped::Twaypoint::staticid = 0;//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
#ifndef _ped_waypoint_h_
#define _ped_waypoint_h_ 1

#ifdef WIN32
#define LIBEXPORT __declspec(dllexport)
#else
#define LIBEXPORT
#endif

#include <cstddef>

using namespace std;

namespace Ped {
class LIBEXPORT Twaypoint {
 public:
  Twaypoint();
  Twaypoint(float x, float y, float r);
  virtual ~Twaypoint();

  // Sets the coordinates and the radius of this waypoint
  void setx(float px) { x = px; };
  void sety(float py) { y = py; };
  void setr(float pr) { r = pr; };

  int getid() const { return id; };
  float getx() const { return x; };
  float gety() const { return y; };
  float getr() const { return r; };

 protected:
  // id incrementer, used for assigning unique ids
  static int staticid;

  // waypoint id
  int id;

  // position of the waypoint
  float x;
  float y;

  // radius defines the area of this waypoint, i.e. a circular
  // area with the middle point in (x,y).
  // Any point within this area is considered to be belonging
  // to this destination.
  float r;
};
}  // namespace Ped

#endif
