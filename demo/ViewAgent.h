///////////////////////////////////////////////////
// Low Level Parallel Programming 2016.
//
//     ==== Don't change this file! ====
//
#ifndef _view_agent_h
#define _view_agent_h

#include <QGraphicsRectItem>
#include <QGraphicsScene>

#include "ped_agents.h"

class ViewAgent {
 public:
  ViewAgent(int agentId, const Ped::Tagents* agents, QGraphicsScene* scene);
  void paint(QColor color);
  const std::pair<int, int> getPosition();

 private:
  int agentIndex;
  const Ped::Tagents* agents;

  // The rectangle on the GUI representing this agent
  QGraphicsRectItem* rect;
  QGraphicsPixmapItem* bgt_icon;
};

#endif
