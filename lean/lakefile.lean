import Lake
open Lake DSL

-- Lean models of the lean4HPC project (core Lean only, no dependencies).
-- Build and check everything:  PATH=<toolchain>/bin:$PATH lake build
package «lean4hpc» where

@[default_target]
lean_lib OptimiKahan where
  globs := #[.submodules `OptimiKahan]
