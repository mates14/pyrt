# SZP Projection Enhancement TODO

## Current Status

SZP (Slant Zenithal Perspective) projection is **implemented but disabled** due to coordinate singularity issues when fitting starts from perpendicular alignment (the optimal mechanical configuration).

## The Problem

**Current parametrization**: `(mu, phi_c, theta_c)`
- `mu`: Distance from projection point to sphere center (controls projection type)
- `phi_c`: Native longitude of tilted reference point (degrees)
- `theta_c`: Native latitude of tilted reference point (degrees)

**Singularity**: When `theta_c = 90°` (perpendicular CCD alignment):
- `phi_c` becomes **undefined/arbitrary** (coordinate singularity at pole)
- This is exactly where most cameras are mechanically aligned!
- Near-perpendicular fits have **exploding Jacobian** - tiny tilts cause phi_c to swing wildly
- Optimizer cannot find gradients → fit fails with NaN/infinite values

**Why it matters**:
- `mu` parameter is valuable: smoothly interpolates TAN → STG → SIN projections
- 2D tilt correction would handle CCD misalignment, tilted field flatteners, decentered optics
- More powerful than ZPN polynomial (no Runge oscillations)

## Proposed Solution

### Reparametrize to Detector-Frame Tilts

**New parameters**: `(mu, tilt_x, tilt_y)`
- `mu`: Distance parameter (same as before)
  - mu = 0 → TAN (gnomonic)
  - mu = 1 → STG (stereographic)
  - mu → ∞ → SIN (orthographic)
- `tilt_x`: Rotation around detector X-axis (degrees) - tilt "up/down" in image frame
- `tilt_y`: Rotation around detector Y-axis (degrees) - tilt "left/right" in image frame

**Advantages**:
- ✅ **No singularity**: Perpendicular = `(mu, 0°, 0°)` - well-defined!
- ✅ **Natural coordinates**: Match mechanical tilt measurements
- ✅ **Small tilts = small parameters**: Good for optimization
- ✅ **Physical interpretation**: Direct mapping to camera/optics alignment

### Implementation Approach

#### 1. Add Conversion Layer in zpnfit.py

```python
def tilt_to_spherical(tilt_x, tilt_y):
    """
    Convert detector-frame tilts to SZP spherical coordinates.

    Args:
        tilt_x: Tilt around detector X-axis (degrees)
        tilt_y: Tilt around detector Y-axis (degrees)

    Returns:
        phi_c: Native longitude (degrees)
        theta_c: Native latitude (degrees)
    """
    tilt_magnitude = np.sqrt(tilt_x**2 + tilt_y**2)

    if tilt_magnitude < 1e-6:
        # Near-perpendicular: phi_c is arbitrary, set to 0
        return 0.0, 90.0

    theta_c = 90.0 - tilt_magnitude
    phi_c = np.degrees(np.arctan2(tilt_y, tilt_x))

    return phi_c, theta_c

def spherical_to_tilt(phi_c, theta_c):
    """
    Convert SZP spherical coordinates to detector-frame tilts.

    Args:
        phi_c: Native longitude (degrees)
        theta_c: Native latitude (degrees)

    Returns:
        tilt_x: Tilt around detector X-axis (degrees)
        tilt_y: Tilt around detector Y-axis (degrees)
    """
    tilt_magnitude = 90.0 - theta_c

    if abs(tilt_magnitude) < 1e-6:
        return 0.0, 0.0

    phi_rad = np.radians(phi_c)
    tilt_x = tilt_magnitude * np.cos(phi_rad)
    tilt_y = tilt_magnitude * np.sin(phi_rad)

    return tilt_x, tilt_y
```

#### 2. Modify SZP Fitting Terms

In `refit_astrometry.py`, instead of fitting PV2_2 and PV2_3 directly:

```python
# Create wrapper parameters TILT_X, TILT_Y
# Map them to PV2_2, PV2_3 via conversion functions
# Fit TILT_X, TILT_Y instead

# Pseudo-code:
zpntest.add_derived_params(
    base_params=['PV2_2', 'PV2_3'],
    derived_params=['TILT_X', 'TILT_Y'],
    forward_transform=tilt_to_spherical,
    inverse_transform=spherical_to_tilt
)

zpntest.fitterm(['PV2_1', 'TILT_X', 'TILT_Y'], [0.0, 0.0, 0.0])
```

#### 3. Update Parameter Handling in termfit.py

May need to add support for **parameter transformations** in the fitting framework:
- Store parameters in transformed space (tilt_x, tilt_y)
- Convert to native space (phi_c, theta_c) before calling projection
- Jacobian transformation for proper error propagation

### Testing Strategy

1. **Synthetic data**: Create images with known tilts, verify recovery
2. **Small tilts**: Start with tilt_x, tilt_y = ±1°, verify convergence
3. **Perpendicular case**: tilt_x = tilt_y = 0, verify no singularity
4. **Combined mu + tilt**: Fit all three parameters simultaneously

### Estimated Complexity

- **Conversion functions**: 30 lines
- **termfit.py parameter transformation support**: 100-150 lines (if needed)
- **Integration into refit_astrometry.py**: 50 lines
- **Testing**: Synthetic test cases

**Total**: ~200-250 lines of code

## Alternative: Use Rotation Matrix Representation

Instead of angle-based parametrization, could use **rotation matrix elements** directly:
- 3 parameters: r11, r12, r13 (first row of rotation matrix)
- Constraints: normalize to unit vector
- Even more stable numerically
- Harder to interpret physically

## Current Workaround

**AZP projection** with `gamma=0` (fixed, no tilt):
- Fits only `mu` parameter
- Smooth interpolation between projection types
- No tilt correction, but avoids polynomial Runge effects
- Good enough for well-aligned cameras

## References

- WCS Paper II (Calabretta & Greisen 2002): SZP definition
- WCSLIB `prj.c`: Reference implementation
- Current implementation: `zpnfit.py` lines 388-407

## Priority

**Medium**: Useful for cameras with mechanical misalignment, but:
- Most cameras are well-aligned (AZP sufficient)
- ZPN polynomial can handle radial distortion
- Multi-image global fitting more urgent for stable camera models

Would be valuable for:
- Cameras with known tilt issues
- Field flattener misalignment diagnosis
- All-sky cameras with wide FOV where tilt matters more
