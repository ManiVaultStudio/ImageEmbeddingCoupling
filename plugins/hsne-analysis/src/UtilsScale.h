#ifndef UTILSSCALE_H
#define UTILSSCALE_H

#include "CommonTypes.h"

#include "PointData/PointData.h"
#include "Dataset.h"

class HsneHierarchy;

namespace mv {
    class Vector2f;
}

namespace utils {

    // Defined in Utils.h, header included in UtilsScale.cpp
    class Vector2D;
    class VisualBudgetRange;
    class VisualTarget;
    class EmbeddingExtends;
    struct ROI;

    enum class POINTINITTYPE : uint32_t {
        previousPos,
        interpolPos,
        randomPos
    };

    constexpr float initTypeToFloat(POINTINITTYPE val) { return static_cast<float>(val); }

    /// ////////////////// ///
    /// HsneScaleFunctions ///
    /// ////////////////// ///
    void extractIdBlock(const utils::Vector2D& roiBottomLeft, const utils::Vector2D& roiTopRight, const Eigen::MatrixXui& imageIndices, std::vector<uint32_t>& idBlock);

    /**  */
    void computeLocalIDsOnRefinedScale(const uint32_t currentScale, const std::vector<uint32_t>& localIDsOnCurrentScale, const HsneHierarchy& hsneHierarchy, const float tresh_influence, std::vector<uint32_t>& localIDsOnRefinedScale);

    /**  */
    void computeLocalIDsOnRefinedScaleHeuristic(const uint32_t currentScale, const std::vector<uint32_t>& localIDsOnCurrentScale, const HsneHierarchy& hsneHierarchy, std::vector<uint32_t>& localIDsOnRefinedScale);

    /** Go bottom up from 0 to newScaleLevel to find the representative landmarks an that scale */
    void computeLocalIDsOnCoarserScale(const uint32_t newScaleLevel, const std::vector<uint32_t>& imageSelectionIDs, const HsneHierarchy& hsneHierarchy, const float tresh_influence, std::vector<uint32_t>& localIDsOnCoarserScale);

    /** Heuristic: Gather the landmark IDs on newScaleLevel that have the highest influence on all of the given data level imageSelectionIDs */
    void computeLocalIDsOnCoarserScaleHeuristic(const uint32_t newScaleLevel, const std::vector<uint32_t>& imageSelectionIDs, const HsneHierarchy& hsneHierarchy, std::vector<uint32_t>& localIDsOnCoarserScale);
    
    /** Wrapper around computeLocalIDsOnCoarserScale{Heuristic}, tresh_influence =-1 will call heuristic, traverses bottom up */
    void localIDsOnCoarserScale(const VisualBudgetRange visualBudget, const std::vector<uint32_t>& imageSelectionIDs, const HsneHierarchy& hsneHierarchy, const float tresh_influence, uint32_t& newScaleLevel, std::vector<uint32_t>& localIDsOnCoarserScale);

    /** Wrapper around computeLocalIDsOnCoarserScale{Heuristic}, tresh_influence =-1 will call heuristic, traverses bottom up */
    void localIDsOnCoarserScale(const VisualTarget visualTarget, const std::vector<uint32_t>& imageSelectionIDs, const HsneHierarchy& hsneHierarchy, const float tresh_influence, uint32_t& newScaleLevel, std::vector<uint32_t>& localIDsOnCoarserScale);
    
    /** Wrapper around computeLocalIDsOnCoarserScale{Heuristic}, tresh_influence =-1 will call heuristic, traverses top down (faster when only zooming in a little) */
    void localIDsOnCoarserScaleTopDown(const VisualBudgetRange visualBudget, const std::vector<uint32_t>& imageSelectionIDs, const HsneHierarchy& hsneHierarchy, const float tresh_influence, uint32_t& newScaleLevel, std::vector<uint32_t>& localIDsOnCoarserScale);

    void landmarkRoiRepresentation(const QSize& imgSize, const utils::ROI& roi, const HsneHierarchy& hsneHierarchy, const uint32_t scaleLevel, const std::vector<uint32_t>& localIDsOnNewScale, std::vector<std::pair<float, std::vector<uint32_t>>>& IdRoiRepresentation);

    void rescaleEmbedding(const mv::Dataset<Points>& embedding, const std::pair<float, float>& embScalingFactors, const utils::EmbeddingExtends& currentEmbExtends, std::vector<mv::Vector2f>& embPosRescaled, utils::EmbeddingExtends& rescaledEmbExtends);

    void reinitializeEmbedding(const HsneHierarchy& hsneHierarchy, const std::vector<mv::Vector2f>& embPositions, const IDMapping& idMap, const utils::EmbeddingExtends& embeddingExtends, const uint32_t newScaleLevel, const std::vector<uint32_t>& localIDsOnCoarserScale, std::vector<float>& initEmbedding, std::vector<utils::POINTINITTYPE>& initTypes);
    
    void recomputeIDMap(const Hsne::Scale& currentScale, const std::vector<uint32_t>& localIDsOnNewScale, IDMapping& idMap);

    /// /// ///
    /// kNN ///
    /// /// ///

    // distance_based_probabilities and neighborhood_graph: size num_dps * num_nn (which is hsne_params._num_neighbors + 1)
    void computeSimilaritiesFromKNN(const std::vector<float>& distance_based_probabilities, const std::vector<uint32_t>& neighborhood_graph, const size_t num_dps, HsneMatrix& similarities);

    /*! Compute exact kNNs
     * Calculate the distances between all point pairs and find closest neighbors
     * \param query_data
     * \param base_data
     * \param num_dps_query
     * \param num_dps_base
     * \param num_dims
     * \param k
     * \param knn_distances_squared
     * \param knn_indices
    */
    void computeExactKNN(const std::vector<float>& query_data, const std::vector<float>& base_data, const size_t num_dps_query, const size_t num_dps_base, const size_t num_dims, const size_t k, std::vector<float>& knn_distances_squared, std::vector<uint32_t>& knn_indices);

    void computeFMC(const size_t num_dps, const size_t nn, std::vector<float>& distance_based_probabilities, std::vector<uint32_t>& knn_indices);

    /// ///////////// ///
    /// HsneHierarchy ///
    /// ///////////// ///

    // modified code from HDILib, https://github.com/biovault/HDILib, MIT Copyright (c) 2017 Nicola Pezzotti
    void extractSubGraph(const HsneMatrix& orig_transition_matrix, const uint32_t threshConnections, std::vector<uint32_t>& selected_idxes, HsneMatrix& new_transition_matrix, float thresh = 0.0f);

}

#endif UTILSSCALE_H