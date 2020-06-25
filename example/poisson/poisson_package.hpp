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
#ifndef EXAMPLE_POISSON_POISSON_PACKAGE_HPP_
#define EXAMPLE_POISSON_POISSON_PACKAGE_HPP_

#include <memory>
#include "interface/container.hpp"
#include "interface/state_descriptor.hpp"
#include "task_list/tasks.hpp"

using parthenon::ParameterInput;
using parthenon::StateDescriptor;

namespace poisson {

  std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
  TaskStatus Smooth(Container<Real> &rc_in, COntainer<Real> &rc_out);

} // namespace poisson_package

#endif // EXAMPLE_POISSON_POISSON_PACKAGE_HPP_
