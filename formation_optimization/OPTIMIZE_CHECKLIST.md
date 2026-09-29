# Spectral-norm search checklist

- [x] Frontier mode: exploit the spectral-norm formation line.
- [x] Optimize submode: loop/search widening.
- [x] Keep topology, planar matrix definition, seed set and matrix metrics fixed.
- [x] Add a direct `min ||T||_2` objective distinct from closest-feasible search.
- [x] Reject the unconstrained `||T||_2=0.8660` candidate because its minimum
  robot spacing is only `0.263 m` and PyBullet terminates immediately.
- [x] Re-sweep position radii 3, 6 and 12 with seeds 0 and 1 and `d_min=0.8 m`.
- [x] Compare best norm, radius, degree conditioning and active bounds.
- [x] Validate the best feasible candidate separately in PyBullet.
- [x] Position-only search has plateaued near `||T||_2 = 1`; none of the
  feasible optima touched the search-box boundary, so widening the box again
  is not the justified next action.
- [x] Next action: change topology/weights or formulate a joint robustness
  objective instead of repeating larger position boxes.
