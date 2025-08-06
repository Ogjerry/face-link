#pragma once

#ifndef NERF_CONFIG_CUH_
#define NERF_CONFIG_CUH_
// This file contains the core architectural parameters for the NeRF model,
// allowing other headers to include it without creating circular dependencies.

// The number of features stored per hash table level.
const int F_val = 4;

// The number of resolution levels in the hash grid.
const int N_LEVELS = 16;

static constexpr int T_per_level = 1 << 19;


static constexpr int N_min = 16;
static constexpr int N_max = 2048;

#endif