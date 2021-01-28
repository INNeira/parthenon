//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#ifndef EXAMPLE_ADVANCED_ADVECTION_ADVANCED_ADVECTION_DRIVER_HPP_
#define EXAMPLE_ADVANCED_ADVECTION_ADVANCED_ADVECTION_DRIVER_HPP_

#include <memory>
#include <vector>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace advanced_advection_example {
using namespace parthenon::driver::prelude;

class AdvancedAdvectionDriver : public MultiStageBlockTaskDriver {
 public:
  AdvancedAdvectionDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm);
  // This next function essentially defines the driver.
  // Call graph looks like
  // main()
  //   EvolutionDriver::Execute (driver.cpp)
  //     MultiStageBlockTaskDriver::Step (multistage.cpp)
  //       DriverUtils::ConstructAndExecuteBlockTasks (driver.hpp)
  //         AdvectionDriver::MakeTaskCollection (advection.cpp)
  TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage);
};

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin);

} // namespace advanced_advection_example

#endif // EXAMPLE_ADVANCED_ADVECTION_ADVANCED_ADVECTION_DRIVER_HPP_
