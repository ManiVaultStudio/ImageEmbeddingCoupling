#include "RegularHsneAction.h"
#include "HsneHierarchy.h"
#include "TsneSettingsAction.h"
#include "DataHierarchyItem.h"
#include "UtilsScale.h"
#include "InteractiveHsnePlugin.h"

#include <QGridLayout>
#include <QMenu>

RegularHsneAction::RegularHsneAction(QObject* parent, TsneSettingsAction& tsneSettingsAction, HsneHierarchy& hsneHierarchy, Dataset<Points> inputDataset, 
    Dataset<Points> embeddingDataset, Dataset<Points> embeddingScatColors, InteractiveHsnePlugin* hsneAnalysisPlugin) :
    GroupAction(parent, "RegularHsneAction", true),
    _tsneSettingsAction(tsneSettingsAction),
    _tsneAnalysis("Regular HSNE"),
    _hsneHierarchy(hsneHierarchy),
    _input(inputDataset),
    _embedding(embeddingDataset),
    _embeddingScatColors(embeddingScatColors),
    _refineEmbedding(),
    _refineEmbScatColors(),
    _refineAction(this, "Refine..."),
    _refineHeuristic(this, "Refine heuristic", false),
    _refineTresh(this, "Refine threshold", 0.0f, 1.0f, 0.5f, 2),
    _recolorAction(this),
    _isTopScale(true), 
    _hsneAnalysisPlugin(hsneAnalysisPlugin),
    _newTransitionMatrix(),
    _currentScaleLevel(0)
{
    setText("HSNE scale");
    setShowLabels(false);

    _refineAction.setToolTip("Refine the selected landmarks");
    _refineHeuristic.setToolTip("Use heuristic to determine refined landmarks for seleciton");
    _refineTresh.setToolTip("Threshold value for precise landmark refinement");

    _refineTresh.setEnabled(true);

    connect(&_refineAction, &TriggerAction::triggered, this, [this]() {
        refine();
        });

    connect(&_refineHeuristic, &ToggleAction::toggled, this, [this](const bool& toggled) {
        _refineTresh.setEnabled(!_refineHeuristic.isChecked());
        });

    const auto updateReadOnly = [this]() -> void {
        auto selection = _embedding->getSelection<Points>();

        _refineAction.setEnabled(!isReadOnly() && !selection->indices.empty() && (_currentScaleLevel > 0));
    };

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
        });

    connect(&_embedding, &Dataset<DatasetImpl>::dataSelectionChanged, this, updateReadOnly);

    updateReadOnly();

    // connect scatter colore recoloring
    connect(&_recolorAction.getColorMapAction(), &ColorMapAction::imageChanged, this, [this](const QImage& image) {
        std::vector<float> scatterColors;
        _hsneAnalysisPlugin->setScatterColorMapData(_embedding, _embeddingScatColors, image, scatterColors);
        });
}

QMenu* RegularHsneAction::getContextMenu(QWidget* parent /*= nullptr*/)
{
    auto menu = new QMenu(text(), parent);

    menu->addAction(&_refineAction);

    return menu;
}

void RegularHsneAction::refine()
{
    utils::ScopedTimer refineTimer("RegularHsneAction::refine total");
    Log::info("Start regular HSNE");

    // Get the selection of points that are to be refined
    auto selection = _embedding->getSelection<Points>();

    // Set the scale of the refined embedding to be one below the current scale
    const auto refinedScaleLevel = _currentScaleLevel - 1;

    // Find proper selection indices
    std::vector<uint32_t> selectedLandmarks; // Selected indices relative to scale
    {
        utils::ScopedTimer refineTimer("RegularHsneAction::get scale relative landmarks");

        std::vector<bool> selectedLocalIndices;
        _embedding->selectedLocalIndices(selection->indices, selectedLocalIndices);

        // Transform local indices to scale relative indices
        for (uint32_t i = 0; i < static_cast<uint32_t>(selectedLocalIndices.size()); i++)
        {
            if (selectedLocalIndices[i])
            {
                selectedLandmarks.push_back(_isTopScale ? i : _drillIndices[i]);
            }
        }
    }

    Log::info(fmt::format("Selected landmarks for regular HSNE: {0} at current scale {1} for new scale {2}", selectedLandmarks.size(), _currentScaleLevel, refinedScaleLevel));

    // Threshold neighbours with enough influence, these represent the indices of the refined points relative to their HSNE scale
    std::vector<uint32_t> refinedLandmarks; // Scale-relative indices

    {
        utils::ScopedTimer landmarkRefinementTimer("RegularHsneAction::landmarkRefinement");

        if(_refineHeuristic.isChecked())
        {
            utils::computeLocalIDsOnRefinedScaleHeuristic(_currentScaleLevel, selectedLandmarks, _hsneHierarchy, refinedLandmarks);
        }
        else
        {
            float tresh = _refineTresh.getValue();
            Log::info(fmt::format("Precise refinement with threshold: {}", tresh));

            // Find the points in the previous level corresponding to selected landmarks
            std::map<uint32_t, float> neighborsRefined;
            _hsneHierarchy.getInfluencedLandmarksInRefinedScale(_currentScaleLevel, selectedLandmarks, neighborsRefined);

            for (auto n : neighborsRefined) {
                if (n.second > tresh) 
                {
                    refinedLandmarks.push_back(n.first);
                }
            }

            Log::info(fmt::format("Landmarks at refined scale: {} (Regular HSNE)", neighborsRefined.size()));
        }
    }

    size_t numRefinedLandmarks = refinedLandmarks.size();

    Log::info(fmt::format("Thresholded landmarks at refined scale: {} (Regular HSNE)", numRefinedLandmarks));
    Log::info("Refining embedding... (Regular HSNE)");

    ////////////////////////////
    // Create refined dataset //
    ////////////////////////////

    // Compute the transition matrix for the landmarks above the threshold
    utils::timer([&]() {
        _hsneHierarchy.getTransitionMatrixForSelectionAtScale(refinedScaleLevel, refinedLandmarks, _newTransitionMatrix);
        },
        "RegularHsneAction::getTransitionMatrixForSelectionAtScale");
    

    // Create a new data set for the embedding
    {
        utils::ScopedTimer landmarkRefinementTimer("RegularHsneAction::createDatasets");

        auto selection = _input->getSelection<Points>();

        Hsne::scale_type& refinedScale = _hsneHierarchy.getScale(refinedScaleLevel);

        selection->indices.clear();

        if (_input->isFull())
        {
            for (size_t i = 0; i < numRefinedLandmarks; i++)
                selection->indices.push_back(refinedScale._landmark_to_original_data_idx[refinedLandmarks[i]]);
        }
        else
        {
            std::vector<unsigned int> globalIndices;
            _input->getGlobalIndices(globalIndices);
            for (size_t i = 0; i < numRefinedLandmarks; i++)
                selection->indices.push_back(globalIndices[refinedScale._landmark_to_original_data_idx[refinedLandmarks[i]]]);
        }

        // Create HSNE scale subset
        auto hsneScaleSubset = _input->createSubsetFromSelection("hsne_scale", _input, false);

        // And the derived data for the embedding
        _refineEmbedding = mv::data().createDerivedDataset<Points>(QString("HSNE Scale (%1)").arg(refinedScaleLevel), hsneScaleSubset, _embedding);
        events().notifyDatasetAdded(_refineEmbedding);

        std::vector<float> initialData(static_cast<size_t>(numRefinedLandmarks) * 2, 0.0f);
        _refineEmbedding->setData(initialData.data(), numRefinedLandmarks, 2);
        events().notifyDatasetDataChanged(_refineEmbedding);

        // Create scatter color data
        _refineEmbScatColors = mv::data().createDerivedDataset("HSNE Scale Scatter Colors", _refineEmbedding, _refineEmbedding);
        events().notifyDatasetAdded(_refineEmbScatColors);

        std::vector<float> scatterColorsTopLevel(numRefinedLandmarks * 3u, 0.0f);
        _refineEmbScatColors->setData(scatterColorsTopLevel.data(), numRefinedLandmarks, 3);
        events().notifyDatasetDataChanged(_refineEmbScatColors);

        _refineEmbedding->getDataHierarchyItem().select();
    }

    // Only add a new scale action if the drill scale is higher than data level
    if (refinedScaleLevel > 0)
    {
        utils::ScopedTimer linkedSelectionTimer("RegularHsneAction::create RegularHsneAction");

        auto regularHsneAction= new RegularHsneAction(this, _tsneSettingsAction, _hsneHierarchy, _input, _refineEmbedding, _refineEmbScatColors, _hsneAnalysisPlugin);
        regularHsneAction->setDrillIndices(refinedLandmarks);
        regularHsneAction->setScale(refinedScaleLevel);

        _refineEmbedding->addAction(*regularHsneAction);
    }

    ///////////////////////////////////
    // Connect scales by linked data //
    ///////////////////////////////////

    // Add linked selection between the refined embedding and the bottom level points
    if (refinedScaleLevel > 0) // Only add a linked selection if it's not the bottom level already
    {
        utils::ScopedTimer linkedSelectionTimer("RegularHsneAction::linked selection");

        LandmarkMap& landmarkMap = _hsneHierarchy.getInfluenceHierarchy().getMapTopDown()[refinedScaleLevel];

        mv::SelectionMap mapping;

        if (_input->isFull())
        {
            for (const unsigned int& scaleIndex : refinedLandmarks)
            {
                int bottomLevelIdx = _hsneHierarchy.getScale(refinedScaleLevel)._landmark_to_original_data_idx[scaleIndex];
                mapping.getMap()[bottomLevelIdx] = landmarkMap[scaleIndex];
            }
        }
        else
        {
            // Link drill-in points to bottom level indices when the original input to HSNE was a subset
            std::vector<unsigned int> globalIndices;
            _input->getGlobalIndices(globalIndices);
            for (const unsigned int& scaleIndex : refinedLandmarks)
            {
                std::vector<unsigned int> bottomMap = landmarkMap[scaleIndex];
                // Transform bottom level indices to the global full set indices
                for (int j = 0; j < bottomMap.size(); j++)
                {
                    bottomMap[j] = globalIndices[bottomMap[j]];
                }
                int bottomLevelIdx = _hsneHierarchy.getScale(refinedScaleLevel)._landmark_to_original_data_idx[scaleIndex];
                mapping.getMap()[globalIndices[bottomLevelIdx]] = bottomMap;
            }
        }

        _refineEmbedding->addLinkedData(_input, mapping);
    }

    _refineEmbedding->getDataHierarchyItem().select();

    // Update embedding points when the TSNE analysis produces new data
    connect(&_tsneAnalysis, &TsneAnalysis::embeddingUpdate, this, [this](const std::vector<float>& emb, const uint32_t& numPoints, const uint32_t& numDimensions) {

        // Update the refine embedding with new data
        _refineEmbedding->setData(emb.data(), numPoints, numDimensions);

        // Notify others that the embedding points have changed
        events().notifyDatasetDataChanged(_refineEmbedding);

        // set scatter colors for regular HSNE embedding
        std::vector<float> scatterColors;
        auto& colorMapAction = _recolorAction.getColorMapAction();
        const auto currentColormap = colorMapAction.getColorMap();
        colorMapAction.setColorMap(currentColormap);
        _hsneAnalysisPlugin->setScatterColorMapData(_refineEmbedding, _refineEmbScatColors, colorMapAction.getColorMapImage(), scatterColors);

        });

    // Start the embedding process
    _tsneAnalysis.startComputation(_tsneSettingsAction.getTsneParameters(), _newTransitionMatrix, static_cast<uint32_t>(numRefinedLandmarks));
}
