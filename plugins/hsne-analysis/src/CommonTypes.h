#pragma once
// some shared type definitions

#include <unordered_map>
#include <map>
#include <vector>
#include <Eigen/Dense>

#pragma warning( push ) 
#pragma warning( disable : 4267 ) // disable 'size_t' to 'uint32_t' warning from external library
#include "hdi/dimensionality_reduction/hierarchical_sne.h"
#pragma warning( pop ) 

#include "hdi/data/map_mem_eff.h"
#include "hdi/data/sparse_mat.h"

struct EmbIdAndPos
{
    uint32_t localIdOnScale;
    uint32_t posInEmbedding;
};

using IDMapping = std::unordered_map<uint32_t, EmbIdAndPos>; // Key -> Data ID, Value -> EmbIdAndPos: localIdOnScale, posInEmbedding. Has localIDsOnNewScale.size() entries, see utils::recomputeIDMap

//using HsneMatrix = std::vector<hdi::data::MapMemEff<uint32_t, float>>;
using HsneMatrix = std::vector<hdi::data::SparseVec<uint32_t, float>>;

using Hsne = hdi::dr::HierarchicalSNE<float, HsneMatrix>;

// A LandmarkMap is nothing but std::vector<std::vector<uint32_t>>
// Example use:
//      The LandmarkMap vector is of the size of numLandmarks on a given scale
//      landmarkMap[i] is a vector of data points (global IDs) on which the landmark i on a given scale (here topScaleIndex) has the highest influence
using LandmarkMap = std::vector<std::vector<uint32_t>>;

using LandmarkMapSingle = std::vector<uint32_t>;

// ID and transision value
using transitionVec = std::vector<std::pair<uint32_t, float>>;

namespace Eigen {
    // Matrix version for uint32_t, works just as MatrixXi
    typedef Matrix<uint32_t, -1, -1> MatrixXui;
}
