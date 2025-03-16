#pragma once

#include "actions/IntegralAction.h"
#include "actions/ToggleAction.h"
#include "actions/OptionAction.h"
#include "actions/DecimalAction.h"
#include <actions/ColorMap1DAction.h>

#include "Dataset.h"
#include "PointData/PointData.h"
#include "ClusterData/ClusterData.h"
#include "util/MeanShift.h"

#include <QRandomGenerator>
#include <memory>

using namespace mv;
using namespace mv::gui;

class QMenu;
class InteractiveHsnePlugin;
class LocalOffscreenBuffer;

/**
 *
 * @author Alexander Vieth
 */
class MeanShiftAction : public GroupAction
{
    Q_OBJECT
public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    MeanShiftAction(QObject* parent, Dataset<Points> topLevelEmb, Dataset<Clusters> embeddingClusters);
    ~MeanShiftAction();

    /** Color options */
    enum class ColorBy {
        PseudoRandomColors,     /** Use pseudo-random colors */
        ColorMap                /** Color by continuous color map */
    };

    void compute();
    void updateColors();

signals:
    void newClusterColors();

public: // Action getters

    DecimalAction& getSigmaAction() { return _sigmaAction; }
    IntegralAction& getNumberClustersAction() { return _numberClustersAction; }
    OptionAction& getColorByAction() { return _colorByAction; }
    ColorMap1DAction& getColorMapAction() { return _colorMapAction; }
    IntegralAction& getRandomSeedAction() { return _randomSeedAction; }
    ToggleAction& getUseClusterColorsAction() { return _useClusterColorsAction; }

protected:
    /// UI
    DecimalAction               _sigmaAction;                   /** Sigma action */
    IntegralAction              _numberClustersAction;          /** Random seed action */
    OptionAction                _colorByAction;                 /** Color by options action */
    ColorMap1DAction            _colorMapAction;                /** Color map action */
    IntegralAction              _randomSeedAction;              /** Random seed action */
    ToggleAction                _useClusterColorsAction;        /** Use cluster colors action */

private:
    /// Functionality
    LocalOffscreenBuffer*               _offscreenBuffer;       /** Off-screen buffer */
    mv::MeanShift                     _meanShift;             /** Mean-shift analysis */
    QRandomGenerator                    _rng;                   /** Random number generator for pseudo-random colors */

    Dataset<Points>                     _embedding;             /** Embedding dataset reference */
    Dataset<Clusters>                   _embeddingClusters;     /** Embedding Clusters dataset reference */
};