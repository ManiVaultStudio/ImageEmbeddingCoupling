#include "UtilsScale.h"

#include "HsneHierarchy.h"
#include "Utils.h"

#include "hdi/utils/math_utils.h"

#include <utility>
#include <algorithm>

#include <cmath>
#include <cstdio>
#include <random>
#include <map>
#include <limits>

namespace utils {
    static float
        L2Sqr(const float* pVect1, const float* pVect2, const size_t qty) {
        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float t = *pVect1 - *pVect2;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return res;
    }
}

namespace utils {

    /// ////////////////// ///
    /// HsneScaleFunctions ///
    /// ////////////////// ///
    void extractIdBlock(const utils::Vector2D& roiBottomLeft, const utils::Vector2D& roiTopRight, const Eigen::MatrixXui& imageIndices, std::vector<uint32_t>& idBlock)
    {
        assert(roiTopRight.x() >= roiBottomLeft.x());
        assert(roiTopRight.y() >= roiBottomLeft.y());

        uint32_t numRows = roiTopRight.x() - roiBottomLeft.x();
        uint32_t numCols = roiTopRight.y() - roiBottomLeft.y();

        Eigen::MatrixXui block = imageIndices.block(roiBottomLeft.x(), roiBottomLeft.y(), numRows, numCols); // startRow, startCol, numRows, numCols
        idBlock.assign(block.data(), block.data() + block.size());
    }

    void computeLocalIDsOnRefinedScale(const uint32_t currentScale, const std::vector<uint32_t>& localIDsOnCurrentScale, const HsneHierarchy& hsneHierarchy, const float tresh_influence, std::vector<uint32_t>& localIDsOnRefinedScale)
    {
        Log::trace(fmt::format("computeLocalIDsOnRefinedScale: newScaleLevel {}", currentScale - 1));

        if (currentScale <= 0)
        {
            Log::error("computeLocalIDsOnRefinedScale: currentScale must be > 0");
            return;
        }

        utils::timer([&]() {
            hsneHierarchy.getLocalIDsInRefinedScale(currentScale, localIDsOnCurrentScale, localIDsOnRefinedScale, tresh_influence);
            },
            "_hsneHierarchy.getLocalIDsInRefinedScale " + std::to_string(currentScale));

    }

    void computeLocalIDsOnRefinedScaleHeuristic(const uint32_t currentScale, const std::vector<uint32_t>& localIDsOnCurrentScale, const HsneHierarchy& hsneHierarchy, std::vector<uint32_t>& localIDsOnRefinedScale)
    {
        auto newScaleLevel = currentScale - 1;
        Log::trace(fmt::format("computeLocalIDsOnRefinedScaleHeuristic: newScaleLevel {}", newScaleLevel));

        if (newScaleLevel < 0)
        {
            Log::error("computeLocalIDsOnRefinedScaleHeuristic: currentScale must be > 0");
            return;
        }

        const std::vector<std::vector<uint32_t>>& influenceMapTopDown = hsneHierarchy.getInfluenceHierarchy().getMapTopDown()[currentScale];

        // get the influenced data points for all selected landmarks
        std::vector<uint32_t> imageSelectionIDs;
        for (const auto& localScaleID : localIDsOnCurrentScale)
        {
            // get the data IDs that are influenced the most by the landmark localScaleID
            const std::vector<uint32_t>& influencedDataIds = influenceMapTopDown[localScaleID];
            imageSelectionIDs.insert(imageSelectionIDs.end(), influencedDataIds.begin(), influencedDataIds.end());
        }

        computeLocalIDsOnCoarserScaleHeuristic(newScaleLevel, imageSelectionIDs, hsneHierarchy, localIDsOnRefinedScale);
    }
    
    void computeLocalIDsOnCoarserScale(const uint32_t newScaleLevel, const std::vector<uint32_t>& imageSelectionIDs, const HsneHierarchy& hsneHierarchy, const float tresh_influence, std::vector<uint32_t>& localIDsOnCoarserScale) 
    {
        Log::trace(fmt::format("computeLocalIDsOnCoarserScale: newScaleLevel {}", newScaleLevel));

        std::vector<uint32_t> localIDsOnScale = imageSelectionIDs;

        for (uint32_t scale_counter = 0; scale_counter < newScaleLevel; scale_counter++)
        {
            utils::timer([&]() {
                hsneHierarchy.getLocalIDsInCoarserScale(scale_counter, localIDsOnScale, localIDsOnCoarserScale, tresh_influence);
                },
                "_hsneHierarchy.getLocalIDsInCoarserScale " + std::to_string(scale_counter));

            localIDsOnScale = localIDsOnCoarserScale;
        }

    }

    void computeLocalIDsOnCoarserScaleHeuristic(const uint32_t newScaleLevel, const std::vector<uint32_t>& imageSelectionIDs, const HsneHierarchy& hsneHierarchy, std::vector<uint32_t>& localIDsOnCoarserScale) 
    {
        Log::trace(fmt::format("computeLocalIDsOnCoarserScaleHeuristic: newScaleLevel {}", newScaleLevel));

        localIDsOnCoarserScale.clear();
        const auto& influenceMapButtomUp = hsneHierarchy.getInfluenceHierarchy().getMapBottomUp()[newScaleLevel];

        // get the influencing landmarks for all selected data points
        for (const auto& imageSelectionID : imageSelectionIDs)
        {
            // get the landmark ID on newScaleLevel that has the highest influence on the data level imageSelectionID (this might be none)
            const std::vector<uint32_t>& influencingLandmarkIds = influenceMapButtomUp[imageSelectionID];
            localIDsOnCoarserScale.insert(localIDsOnCoarserScale.end(), influencingLandmarkIds.begin(), influencingLandmarkIds.end());
        }

        // only retain the unique IDs: sort, unique, erase, see https://en.cppreference.com/w/cpp/algorithm/unique
        std::sort(utils::exec_policy, localIDsOnCoarserScale.begin(), localIDsOnCoarserScale.end());
        auto last = std::unique(utils::exec_policy, localIDsOnCoarserScale.begin(), localIDsOnCoarserScale.end());
        localIDsOnCoarserScale.erase(last, localIDsOnCoarserScale.end());
    }

    void localIDsOnCoarserScale(const VisualBudgetRange visualBudget, const std::vector<uint32_t>& imageSelectionIDs, const HsneHierarchy& hsneHierarchy, const float tresh_influence, uint32_t& newScaleLevel, std::vector<uint32_t>& localIDsOnCoarserScale) {

        Log::info(fmt::format("localIDsOnCoarserScale: Visual budget max: {0}", visualBudget.getMax()));
        Log::info(fmt::format("localIDsOnCoarserScale: Use influence {}", (tresh_influence == -1.0f) ? "heuristic" : "threshold of " + std::to_string(tresh_influence)));

        // go up the scales until the number of landmarks falls below the max visual budget
        uint32_t levelCounter = 0;

        // skip data level scale if num selected data level points is larger than visual budget
        if ((imageSelectionIDs.size() > visualBudget.getMax()) && (hsneHierarchy.getNumScales() > 1))
        {
            localIDsOnCoarserScale = imageSelectionIDs;
            levelCounter = 1;
        }

        // go to top level when entire image is in view
        if (imageSelectionIDs.size() >= hsneHierarchy.getNumPoints())
        {
            localIDsOnCoarserScale = imageSelectionIDs;
            levelCounter = hsneHierarchy.getTopScale();
        }

        for(levelCounter; levelCounter < hsneHierarchy.getNumScales(); levelCounter++)
        {
            Log::trace(fmt::format("localIDsOnCoarserScale: newScaleLevel {}", levelCounter));

            if (tresh_influence == -1.0f)
            {
                computeLocalIDsOnCoarserScaleHeuristic(levelCounter, imageSelectionIDs, hsneHierarchy, localIDsOnCoarserScale);
            }
            else
            {
                if(levelCounter == 0)
                    hsneHierarchy.getLocalIDsInCoarserScale(levelCounter, imageSelectionIDs, localIDsOnCoarserScale, tresh_influence);
                else
                    hsneHierarchy.getLocalIDsInCoarserScale(levelCounter, localIDsOnCoarserScale, localIDsOnCoarserScale, tresh_influence);
            }

            Log::info("computeLocalIDsOnCoarserScaleHeuristic: " + std::to_string(localIDsOnCoarserScale.size()) + " landmarks on scale " + std::to_string(levelCounter));

            newScaleLevel = levelCounter;

            // if number of IDs on newScaleLevel is within the visual budget range, stay on newScaleLevel
            if (localIDsOnCoarserScale.size() < visualBudget.getMax())
                break;
        }
    }

    void localIDsOnCoarserScale(const VisualTarget visualTarget, const std::vector<uint32_t>& imageSelectionIDs, const HsneHierarchy& hsneHierarchy, const float tresh_influence, uint32_t& newScaleLevel, std::vector<uint32_t>& localIDsOnCoarserScale) {

        Log::info(fmt::format("localIDsOnCoarserScale: Visual target: {0}", visualTarget.getTarget()));
        Log::info(fmt::format("localIDsOnCoarserScale: Use influence {0}", (tresh_influence == -1.0f) ? "heuristic" : "threshold of " + std::to_string(tresh_influence)));

        const auto topScale = hsneHierarchy.getTopScale();
        const auto numPoints = hsneHierarchy.getNumPoints();
        const auto numSelection = imageSelectionIDs.size();
        const auto target = visualTarget.getTarget();

        // traverse scales until the number of landmarks falls below the max visual budget
        uint32_t levelCounter = 0;

        // cache 
        std::vector<uint32_t> cache_localIDsOnCoarserScale;

        // set direction identifier
        const bool UP = true;
        const bool DOWN = false;
        bool traverseDirection = UP;

        // levelCounter updates
        auto cComp = +[](uint32_t counter, uint32_t comp) -> bool { return counter <= comp; };
        auto cUpdate = +[](uint32_t& counter) -> void { counter++; };

        // skip data level scale if num data level points is larger than visual target
        if ((numSelection > (10u * target)) && (hsneHierarchy.getNumScales() > 1))
        {
            localIDsOnCoarserScale = imageSelectionIDs;
            levelCounter = 1;
        }

        // go to top level when entire image is in view
        if (numSelection >= numPoints)
        {
            localIDsOnCoarserScale = imageSelectionIDs;
            levelCounter = topScale;
        }

        // check if traverse down instead of up
        if (visualTarget.getHeuristic())
        {
            if (numSelection > 0.125f * numPoints)
                traverseDirection = DOWN;

            if (traverseDirection == DOWN)
            {
                cComp = +[](uint32_t counter, uint32_t comp) -> bool { return counter >= 0 && counter < 10000; };   // prevent wrap-around of uint
                cUpdate = +[](uint32_t& counter) -> void { counter--; };
                levelCounter = topScale;
                localIDsOnCoarserScale = imageSelectionIDs;
            }
        }

        Log::info(fmt::format("localIDsOnCoarserScale: traverseDirection: {0}", traverseDirection ? "UP" : "DOWN"));

        // traverse hierarchy
        for (levelCounter; cComp(levelCounter, topScale); cUpdate(levelCounter))
        {
            Log::debug(fmt::format("localIDsOnCoarserScale: newScaleLevel {0}", levelCounter));
            cache_localIDsOnCoarserScale = localIDsOnCoarserScale;

            if (tresh_influence == -1.0f)
            {
                computeLocalIDsOnCoarserScaleHeuristic(levelCounter, imageSelectionIDs, hsneHierarchy, localIDsOnCoarserScale);
            }
            else
            {
                if (levelCounter == 0)
                    hsneHierarchy.getLocalIDsInCoarserScale(levelCounter, imageSelectionIDs, localIDsOnCoarserScale, tresh_influence);
                else
                    hsneHierarchy.getLocalIDsInCoarserScale(levelCounter, localIDsOnCoarserScale, localIDsOnCoarserScale, tresh_influence);
            }

            Log::info("localIDsOnCoarserScale: " + std::to_string(localIDsOnCoarserScale.size()) + " landmarks on scale " + std::to_string(levelCounter));

            if (traverseDirection == UP)
            {
                // if number of IDs on newScaleLevel is smaller than the visual target, the next upper scale cannot be closer (since there will be even fewer points)
                if (localIDsOnCoarserScale.size() <= target)
                    break;
            }

            // reverse of above, also enforce top level when all image points are in view
            if (traverseDirection == DOWN)
            {
                if ((localIDsOnCoarserScale.size() > target) || (numSelection >= numPoints))
                    break;
            }
        }

        // Check edge-case wrap around
        if((traverseDirection == DOWN) && levelCounter == std::numeric_limits<uint32_t>::max())
            newScaleLevel = 0;
        else
            newScaleLevel = levelCounter;

        if ((traverseDirection == UP) && newScaleLevel == 0)
            return;

        auto abs_diff = [](size_t a, size_t b) -> int32_t {
            return std::abs(static_cast<int32_t>(a) - static_cast<int32_t>(b));
        };

        if (abs_diff(cache_localIDsOnCoarserScale.size(), target) < abs_diff(localIDsOnCoarserScale.size(), target))
        {
            if ((traverseDirection == UP) && (newScaleLevel > 0))
                newScaleLevel--;
            else
                newScaleLevel++;

            std::swap(localIDsOnCoarserScale, cache_localIDsOnCoarserScale);
        }

    }

    void localIDsOnCoarserScaleTopDown(const VisualBudgetRange visualBudget, const std::vector<uint32_t>& imageSelectionIDs, const HsneHierarchy& hsneHierarchy, const float tresh_influence, uint32_t& newScaleLevel, std::vector<uint32_t>& localIDsOnCoarserScale) {

        Log::info(fmt::format("localIDsOnCoarserScaleTopDown: Visual range: [{0}, {1}]", visualBudget.getMin(), visualBudget.getMax()));
        Log::info(fmt::format("localIDsOnCoarserScaleTopDown: Use {}", (tresh_influence == -1.0f) ? "heuristic" : "threshold of " + std::to_string(tresh_influence)));

        // cache in case we have to use newScaleLevel-- for newScaleLevel exceeding visualBudget range (then newScaleLevel being lower visualBudget.Min is ok)
        std::vector<uint32_t> cache_localIDsOnCoarserScale;

        // continue to go down the scales until the number of landmarks falls within the predefined visual budget
        // we start on the top scale, given a selection we traverse the hierarchy downwards
        for (uint32_t levelCounter = hsneHierarchy.getTopScale(); levelCounter != static_cast<uint32_t>(-1); levelCounter--)
        {
            newScaleLevel = levelCounter;
            cache_localIDsOnCoarserScale = localIDsOnCoarserScale;

            if (tresh_influence == -1.0f)
                computeLocalIDsOnCoarserScaleHeuristic(newScaleLevel, imageSelectionIDs, hsneHierarchy, localIDsOnCoarserScale);
            else
                computeLocalIDsOnCoarserScale(newScaleLevel, imageSelectionIDs, hsneHierarchy, tresh_influence, localIDsOnCoarserScale);

            Log::info("localIDsOnCoarserScaleTopDown: " + std::to_string(localIDsOnCoarserScale.size()) + " landmarks on scale " + std::to_string(newScaleLevel));

            // Using the visual budget heuristic:
            // if number of IDs on newScaleLevel is within the visual budget range, stay here
            if (visualBudget.getHeuristic() && visualBudget.isWithinRange(localIDsOnCoarserScale.size()))
                break;

            // if that number is larger than the budget, use previous level (except for top scale)
            if (localIDsOnCoarserScale.size() > visualBudget.getMax())
            {
                if (newScaleLevel != hsneHierarchy.getTopScale())
                {
                    newScaleLevel++;
                    localIDsOnCoarserScale = cache_localIDsOnCoarserScale;
                }

                break;
            }
            // otherwise continue on more detailed scale
        }
    }

    void landmarkRoiRepresentation(const QSize& imgSize, const utils::ROI& roi, const HsneHierarchy& hsneHierarchy, const uint32_t scaleLevel, const std::vector<uint32_t>& localIDsOnScale, std::vector<std::pair<float, std::vector<uint32_t>>>& IdRoiRepresentation)
    {
        IdRoiRepresentation.resize(localIDsOnScale.size());

        // Data level landmarks only not represent themself
        if (scaleLevel == 0)
        {
            for (uint32_t i = 0; i < localIDsOnScale.size(); i++)
                IdRoiRepresentation[i] = { 1, {localIDsOnScale[i]} };
            return;
        }

        // _influenceMapTopDown[landmarkIDOnScale]: vector of data point IDs for which landmarkIDOnScale has the highest influence on
        const auto& influenceMapTopDown = hsneHierarchy.getInfluenceHierarchy().getMapTopDown()[scaleLevel];
        //const auto& scale = hsneHierarchy.getScale(scaleLevel); // scale._landmark_to_original_data_idx[emdId]

        auto getPixelCoordinateFromPixelIndex = [&imgSize](const std::int32_t& pixelIndex) -> QPoint {
            return QPoint(pixelIndex % imgSize.width(), static_cast<std::int32_t>(pixelIndex / static_cast<float>(imgSize.width())));
        };

        auto range = pyrange(localIDsOnScale.size());
        std::for_each(range.begin(), range.end(), [&](auto i) {
            // TODO: use exact representation instead of Heuristic
            const uint32_t id = localIDsOnScale[i];
     
            // vector of data point IDs for which landmarkIDOnScale has the highest influence on
            const std::vector<uint32_t>& influencedDataPoints = influenceMapTopDown[id];

            IdRoiRepresentation[i].second.clear();
            for (const uint32_t& influencedDataPoint : influencedDataPoints)
            {
                QPoint pixel = getPixelCoordinateFromPixelIndex(influencedDataPoint);
                if (pixelInRoi(pixel.x(), pixel.y(), roi))
                    IdRoiRepresentation[i].second.push_back(influencedDataPoint);
            }

            if (influencedDataPoints.size() > 0)
                IdRoiRepresentation[i].first = static_cast<float>(IdRoiRepresentation[i].second.size()) / static_cast<float>(influencedDataPoints.size());
            else
                IdRoiRepresentation[i].first = 0;

        });
    }

    void rescaleEmbedding(const mv::Dataset<Points>& embedding, const std::pair<float, float>& embScalingFactors, const utils::EmbeddingExtends& currentEmbExtends,
        std::vector<mv::Vector2f>& embPosRescaled, utils::EmbeddingExtends& rescaledEmbExtends)
    {
        embedding->extractDataForDimensions(embPosRescaled, 0, 1);

        const mv::Vector2f scaleFact = { embScalingFactors.first, embScalingFactors.second };

        Log::info(fmt::format("rescaleEmbedding: Rescale factor: scaleX {0}, scaleY {1}", embScalingFactors.first, embScalingFactors.second));

        // rescale embedding
        auto range = utils::pyrange(embPosRescaled.size());
        std::for_each(utils::exec_policy, range.begin(), range.end(), [&](const auto i) {
            embPosRescaled[i] *= scaleFact;
            });

        // rescale embedding
        rescaledEmbExtends.setExtends(currentEmbExtends.x_min() * scaleFact.x, currentEmbExtends.x_max() * scaleFact.x,
                                      currentEmbExtends.y_min() * scaleFact.y, currentEmbExtends.y_max() * scaleFact.y);

        Log::debug("currentEmbedding: Embedding extends (before rescale): " + currentEmbExtends.getMinMaxString());
        Log::debug("rescaleEmbedding: Embedding extends (after rescale): " + rescaledEmbExtends.getMinMaxString());
    }

    void reinitializeEmbedding(const HsneHierarchy& hsneHierarchy, const std::vector<mv::Vector2f>& embPositions, const IDMapping& idMap, const utils::EmbeddingExtends& embeddingExtends,
        const uint32_t newScaleLevel, const std::vector<uint32_t>& localIDsOnNewScale, std::vector<float>& initEmbedding, std::vector<utils::POINTINITTYPE>& initTypes)
    {
        // resize embedding positions and meta into vectors
        initEmbedding.resize(localIDsOnNewScale.size() * 2);
        initTypes.resize(localIDsOnNewScale.size());

        // compute max radii for random init
        assert(embeddingExtends.extend_x() > 0 && embeddingExtends.extend_y() > 0);
        const float rad_randomMax_X = std::max(std::abs(embeddingExtends.x_min()), std::abs(embeddingExtends.x_max()));
        const float rad_randomMax_Y = std::max(std::abs(embeddingExtends.y_min()), std::abs(embeddingExtends.y_max()));

        // some access helper
        const auto& newScale = hsneHierarchy.getScale(newScaleLevel);
        const auto& transitionNNsOnScale = hsneHierarchy.getTransitionNNOnScale(newScaleLevel);
        const HsneMatrix& fullTransitionMatrix = newScale._transition_matrix;

        // debug and logging counters
        size_t numPoints_oldPos(0), numPoints_interPos(0), numPoints_randPos(0);
        Log::info("reinitializeEmbedding:: Old embedding size of " + std::to_string(embPositions.size()) + " and new size of " + std::to_string(localIDsOnNewScale.size()));
        Log::info("reinitializeEmbedding:: Random init max radii (x, y): " + std::to_string(rad_randomMax_X) + ", " + std::to_string(rad_randomMax_Y));

        // for each point in the new embedding, either 
        //  1) initialize it at it's old position if the landmark has been in the previous embedding
        //  2) interpolate it's new position based on nearest (transition) landmark neighbors
        //  3) use a random position
        for (size_t emdId = 0; emdId < localIDsOnNewScale.size(); emdId++)
        {
            const auto embId_x = 2u * emdId + 0u;
            const auto embId_y = 2u * emdId + 1u;

            const auto idMapEntryCurrentPoint = idMap.find(newScale._landmark_to_original_data_idx[localIDsOnNewScale[emdId]]);

            // Use the embedding position if the landmark was already in the embedding
            if (idMapEntryCurrentPoint != idMap.end())
            {
                // idMapEntry stores the localIDs on the previous scale since idMap is recomputed later
                // localIdOnPreviousScale = idMapEntry->second.posInEmbedding
                const auto& previousPoint = embPositions[idMapEntryCurrentPoint->second.posInEmbedding];

                initEmbedding[embId_x] = previousPoint.x;
                initEmbedding[embId_y] = previousPoint.y;

                initTypes[emdId] = POINTINITTYPE::previousPos;
                numPoints_oldPos++;
            }
            else
            {
                // If the landmark was not present in the previous embedding, use a heuristic based on nearest neighbors
                // Take the nearest neighbors (wrt to the transition matrix) of landmark emdId on a given scale
                // Take the three nearest of them which have been in the previous embedding
                // Use the center of their embedding positions as the new init value
                // If none of the nearest neighbors were in the previous embedding, use random values

                constexpr uint32_t numN = 3;
                std::array<uint32_t, numN> transitNeighborsEmbPos = { 0, 0, 0 };

                bool interpolatePos = false;
                uint32_t nnCount = 0;

                // type of transitionNNsOnScale (for a given scale) is 
                // std::vector<std::vector<uint32_t>>
                //    landmarks      kNN      ID (local on scale)
                // type of transitionNNs is
                // std::vector<uint32_t>
                //         kNN     ID (local on scale)

                // get the sorted transition values (and respective locas IDs)
                const std::vector<uint32_t>& transitionNNs = transitionNNsOnScale[localIDsOnNewScale[emdId]];

                // check whether any of the transition landmarks have been in the previous embedding
                for (const auto& transitID : transitionNNs)
                {
                    const auto idMapEntryTransitNeighbor = idMap.find(newScale._landmark_to_original_data_idx[transitID]);
                    if (idMapEntryTransitNeighbor != idMap.end())
                    {
                        transitNeighborsEmbPos[nnCount++] = idMapEntryTransitNeighbor->second.posInEmbedding;

                        // when three transition neighbors have been in the previous embedding, use them for interpolation
                        if (nnCount == numN)
                        {
                            interpolatePos = true;
                            break;
                        }
                    }
                }

                if (interpolatePos)
                {
                    // Interpolate with trantision neighbors' positions
                    assert(!std::equal(transitNeighborsEmbPos.begin() + 1, transitNeighborsEmbPos.end(), transitNeighborsEmbPos.begin())); // check that all neigbors are different

                    auto interpolPoint = utils::interpol2D(embPositions[transitNeighborsEmbPos[0]], embPositions[transitNeighborsEmbPos[1]], embPositions[transitNeighborsEmbPos[2]]);
                    
#ifndef NDEBUG
                    // In debug, warn if a point is not interpolated in the triangle spanned by the three base points
                    // This happens e.g. when two base points lie on top of each other
                    if (!utils::pointInTriangle(interpolPoint, embPositions[transitNeighborsEmbPos[0]], embPositions[transitNeighborsEmbPos[1]], embPositions[transitNeighborsEmbPos[2]]))
                    {
                        Log::warn(fmt::format("({0}, {1}) not in [({2}, {3}), ({4}, {5}), ({6}, {7})]", interpolPoint.x, interpolPoint.y,
                            embPositions[transitNeighborsEmbPos[0]].x, embPositions[transitNeighborsEmbPos[0]].y,
                            embPositions[transitNeighborsEmbPos[1]].x, embPositions[transitNeighborsEmbPos[1]].y,
                            embPositions[transitNeighborsEmbPos[2]].x, embPositions[transitNeighborsEmbPos[2]].y
                            ));
                    }
#endif

                    initEmbedding[embId_x] = interpolPoint.x;
                    initEmbedding[embId_y] = interpolPoint.y;

                    initTypes[emdId] = POINTINITTYPE::interpolPos;
                    numPoints_interPos++;
                }
                else
                {
                    // Last resort: use a random position 
                    auto randomPoint = utils::randomVec(rad_randomMax_X, rad_randomMax_Y);

                    initEmbedding[embId_x] = randomPoint.x;
                    initEmbedding[embId_y] = randomPoint.y;

                    initTypes[emdId] = POINTINITTYPE::randomPos;
                    numPoints_randPos++;
                }
            }
        }

        assert(numPoints_oldPos + numPoints_interPos + numPoints_randPos == localIDsOnNewScale.size());

        Log::info("reinitializeEmbedding:: Old pos " + std::to_string(numPoints_oldPos) + ", interpol pos " + std::to_string(numPoints_interPos) + 
            ", rand pos " + std::to_string(numPoints_randPos) + " of total " + std::to_string(localIDsOnNewScale.size()) + " (" + std::to_string(numPoints_oldPos + numPoints_interPos + numPoints_randPos) + ")");
    }

    void recomputeIDMap(const Hsne::Scale& currentScale, const std::vector<uint32_t>& localIDsOnNewScale, IDMapping& idMap)
    {
        idMap.clear();
        idMap.max_load_factor(1);                   // Make sure that there is only one elements per bucket
        idMap.reserve(localIDsOnNewScale.size());   // Prevent rehashing when inserting values

        for (uint32_t i = 0; i < localIDsOnNewScale.size(); i++)
        {
            // Key -> Data ID, Value -> EmbIdAndPos: localIdOnScale, posInEmbedding
            idMap.insert({ currentScale._landmark_to_original_data_idx[localIDsOnNewScale[i]],  // Key -> Data ID
                         { localIDsOnNewScale[i] , i }                                          // Value -> EmbIdAndPos: localIdOnScale, posInEmbedding
                });
        }
    }

    void computeSimilaritiesFromKNN(const std::vector<float>& distance_based_probabilities, const std::vector<uint32_t>& neighborhood_graph, const size_t num_dps, HsneMatrix& similarities)
    {
        Log::info("computeSimilaritiesFromKNN from knn");

        similarities.resize(num_dps);

        size_t nn = distance_based_probabilities.size() / num_dps;

        // same as in hdilib\hdi\dimensionality_reduction\hierarchical_sne_inl.h: initializeFirstScale()

        auto range = utils::pyrange(num_dps);
        std::for_each(utils::exec_policy, range.begin(), range.end(), [&](const auto i) {
            float sum = 0;
            for (size_t n = 1; n < nn; ++n) {
                size_t idx = i * nn + n;
                auto v = distance_based_probabilities[idx];
                sum += v;
                similarities[i][neighborhood_graph[idx]] = v;
            }
            });

        Log::info("computeSimilaritiesFromKNN from knn finished");

    }

    void computeExactKNN(const std::vector<float>& query_data, const std::vector<float>& base_data, const size_t num_dps_query, const size_t num_dps_base, const size_t num_dims, const size_t k, std::vector<float>& knn_distances_squared, std::vector<uint32_t>& knn_indices)
    {
        knn_distances_squared.resize(num_dps_query * k, -1.0f);
        knn_indices.resize(num_dps_query * k, -1);

        Log::info("computeExactKNN");

        std::vector<std::pair<int, float>> indices_distances(num_dps_query);

        // For each point, calc distances to all other
        // and take the nn smallest as kNN
        for (int i = 0; i < static_cast<int>(num_dps_query); i++) {
            // Calculate distance to all points  using the respective metric
            #ifdef NDEBUG
            #pragma omp parallel for
            #endif
            for (int j = 0; j < static_cast<int>(num_dps_query); j++) 
                indices_distances[j] = std::make_pair(j, L2Sqr(query_data.data() + i * num_dims, query_data.data() + j * num_dims, num_dims));

            // sort all distances to point i
            std::sort(indices_distances.begin(), indices_distances.end(), [](std::pair<int, float> a, std::pair<int, float> b) {return a.second < b.second; });

            // Take the first nn indices 
            std::transform(indices_distances.begin(), indices_distances.begin() + k, knn_indices.begin() + i * k, [](const std::pair<int, float>& p) { return p.first; });

            // Take the first nn distances 
            std::transform(indices_distances.begin(), indices_distances.begin() + k, knn_distances_squared.begin() + i * k, [](const std::pair<int, float>& p) { return p.second; });
        }

    }

    void computeFMC(const size_t num_dps, const size_t nn, std::vector<float>& distance_based_probabilities, std::vector<uint32_t>& knn_indices)
    {
        Log::info("FMC computation");

        // same part of hdilib\hdi\dimensionality_reduction\hierarchical_sne_inl.h: computeNeighborhoodGraph()

        float perplexity = nn / 3.f;

        auto range = utils::pyrange(num_dps);
        std::for_each(utils::exec_policy, range.begin(), range.end(), [&](const auto d) {
            //It could be that the point itself is not the nearest one if two points are identical... I want the point itself to be the first one!
            if (knn_indices[d * nn] != d) {
                size_t to_swap = d * nn;
                for (; to_swap < d * nn + (nn - 1); ++to_swap) {
                    if (knn_indices[to_swap] == d)
                        break;
                }
                std::swap(knn_indices[nn * d], knn_indices[to_swap]);
                std::swap(distance_based_probabilities[nn * d], distance_based_probabilities[to_swap]);
            }
            std::vector<float> temp_probability(nn, 0);
            ::hdi::utils::computeGaussianDistributionWithFixedPerplexity<std::vector<float>>(
                distance_based_probabilities.cbegin() + d * nn,
                distance_based_probabilities.cbegin() + (d + 1) * nn,
                temp_probability.begin(),
                temp_probability.begin() + nn,
                perplexity,
                200,
                1e-5,
                0
                );
            distance_based_probabilities[d * nn] = 0;
            for (size_t n = 1; n < nn; ++n) {
                distance_based_probabilities[d * nn + n] = temp_probability[n];
            }
        });
        
        Log::info("FMC computation finished");

    }

    void extractSubGraph(const HsneMatrix& orig_transition_matrix, const uint32_t threshConnections, std::vector<uint32_t>& selected_idxes, HsneMatrix& new_transition_matrix, float thresh)
    {
        // Difference to HDILib version: Here, we don't add vertices that are connected to a selected vertex since we already now all new scale landmarks for which we want connections
        //                               Adding also landmarks that are connected to those, would add unwanted landmarks to the selection
        new_transition_matrix.clear();
        new_transition_matrix.resize(selected_idxes.size());

        constexpr uint32_t NOTFOUND = std::numeric_limits<uint32_t>::max();

        std::vector<uint32_t> map_selected_idxes;
        map_selected_idxes.resize(std::max(orig_transition_matrix.size(), static_cast<size_t>(selected_idxes[selected_idxes.size()-1]+1)), NOTFOUND);     // Favor speed over memory

        // The selected rows must be taken completely
        uint32_t newIdCounter = 0;
        for (const uint32_t& id : selected_idxes) {
            map_selected_idxes[id] = newIdCounter++;
        }

        // Resize Eigen sparse matrix
        for (auto& row : new_transition_matrix)
            row.resize(newIdCounter);

        // Now that I have the maps, I generate the new transition matrix
        // we can parallelize this since map_selected_idxes[selected_idxes[i]] will be a different number for each i
        auto rangeSel = utils::pyrange(selected_idxes.size());
        std::for_each(utils::exec_policy, rangeSel.begin(), rangeSel.end(), [&](auto i) {
            for (Eigen::SparseVector<float>::InnerIterator it(orig_transition_matrix[selected_idxes[i]].memory()); it; ++it) {
                if ((map_selected_idxes[it.index()] != NOTFOUND) && (it.value() > thresh)) {
                    new_transition_matrix[map_selected_idxes[selected_idxes[i]]][map_selected_idxes[it.index()]] = it.value();
                }
            }
        });

        // Filter entries with fewer than threshConnections connections
        if (threshConnections > 0)
        {            
            utils::ScopedTimer filterLandMarkTimer("Filter landmarks with low number of connections");

            std::vector<uint32_t> valid_vertices;
            std::vector<uint32_t> invalid_vertices;

            for (uint32_t row = 0; row < new_transition_matrix.size(); row++)
            {
                if (new_transition_matrix[row].size() >= threshConnections)
                    valid_vertices.push_back(row);
                else
                    invalid_vertices.push_back(row);
            }

            Log::info(fmt::format("extractSubGraph, remove {0} landmarks with fewer than {1} transitions", invalid_vertices.size(), threshConnections));

            // Remove IDs at positions invalid_vertices from selected_idxes and new_transition_matrix
            utils::eraseElements(selected_idxes, invalid_vertices);
            utils::eraseElements(new_transition_matrix, invalid_vertices);

            // Remove connections to the removed invalid_vertices
            //hdi::utils::removeEdgesToUnselectedVertices(new_transition_matrix, valid_vertices);
            {   // equvalent but marginally faster than the above by avoiding unordered_map in favor of using some more memory, and due to parallelization
                std::vector<uint32_t> valid_set;
                valid_set.resize(valid_vertices.size() + invalid_vertices.size(), NOTFOUND);

                auto rangeVert = utils::pyrange(static_cast<uint32_t>(valid_vertices.size()));
                std::for_each(utils::exec_policy, rangeVert.begin(), rangeVert.end(), [&](auto i) {
                    valid_set[valid_vertices[i]] = i;
                });

                HsneMatrix new_map(new_transition_matrix.size());
                auto rangeTrans = utils::pyrange(valid_vertices.size());
                std::for_each(utils::exec_policy, rangeTrans.begin(), rangeTrans.end(), [&](auto i) {
                    //for (auto& elem : new_transition_matrix[i]) {
                    for (Eigen::SparseVector<float>::InnerIterator it(new_transition_matrix[i].memory()); it; ++it) {
                        if (valid_set[it.index()] != NOTFOUND) {
                            new_map[i][valid_set[it.index()]] = it.value();
                        }
                    }
                });

                new_transition_matrix = new_map;
            }
        }

        //Finally, the new transition matrix must be normalized
        double sum = 0;
        for (auto& row : new_transition_matrix) {
            //for (auto& elem : row) {
            for (Eigen::SparseVector<float>::InnerIterator it(row.memory()); it; ++it) {
                sum += it.value();
            }
        }
        for (auto& row : new_transition_matrix) {
            for (Eigen::SparseVector<float>::InnerIterator it(row.memory()); it; ++it) {
                it.valueRef() = new_transition_matrix.size() * it.value() / sum;
            }
        }

    }

}