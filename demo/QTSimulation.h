#ifndef _qt_simulation_h_
#define _qt_simulation_h_

#include <QObject>
#include <QTimer>

#include "MainWindow.h"
#include "Simulation.h"

class QTSimulation : public QObject, public Simulation {
  Q_OBJECT

 public:
  QTSimulation(Ped::Model &model, int maxSteps, MainWindow *window,
               QObject *parent = nullptr);
  QTSimulation(Ped::Model &model, int maxSteps);
  QTSimulation() = delete;
  ~QTSimulation() {}

  void runSimulation();
 public slots:
  void simulateOneStep();

 protected:
  QTimer movetimer;
  MainWindow *window;
};

#endif
