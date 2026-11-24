/- UNCHECKED DRAFT: Lean is unavailable on the current machine.
   Concrete integer arithmetic witness for the extracted sequential update.
   This is NOT a formal CUDA/IEEE-754 semantics or source refinement proof.
   The witness uses only small exactly representable integers.
   Intended command: lean ComplexBeta.lean (Lean 4 core, no Mathlib).
-/
def sourceUpdate (x y br bi : Int) : Int × Int :=
  let newX := x * br - y * bi
  (newX, y * br + newX * bi)

def referenceUpdate (x y br bi : Int) : Int × Int :=
  (x * br - y * bi, y * br + x * bi)

theorem sourceWitness : sourceUpdate 1 2 3 4 = (-5, -14) := by decide
theorem referenceWitness : referenceUpdate 1 2 3 4 = (-5, 10) := by decide
theorem mismatch : sourceUpdate 1 2 3 4 ≠ referenceUpdate 1 2 3 4 := by decide
