#include "MeanShiftAction.h"
#include "DataHierarchyItem.h"

#include "Logger.h"

#include <QWindow>
#include <QOpenGLContext>

class LocalOffscreenBuffer : public QWindow
{
public:
    LocalOffscreenBuffer()
    {
        setSurfaceType(QWindow::OpenGLSurface);

        _context = new QOpenGLContext(this);
        _context->setFormat(requestedFormat());

        if (!_context->create())
            qFatal("Cannot create requested OpenGL context.");

        create();
    }

    QOpenGLContext* getContext() { return _context; }

    void bindContext()
    {
        _context->makeCurrent(this);
    }

    void releaseContext()
    {
        _context->doneCurrent();
    }

private:
    QOpenGLContext* _context;
};


using namespace mv::gui;
using namespace mv;

MeanShiftAction::MeanShiftAction(QObject* parent, Dataset<Points> topLevelEmb, Dataset<Clusters> embeddingClusters) :
    GroupAction(parent, "MeanShiftAction", false),
    _meanShift(),
    _offscreenBuffer(nullptr),
    _sigmaAction(this, "Sigma", 0.01f, 3.0f, 0.15f, 3),
    _numberClustersAction(this, "Number clusters", 0, 1000,  1),
    _colorByAction(this, "Color by", QStringList({ "Pseudo-random colors", "Color map" }), "Color map"),
    _colorMapAction(this, "Color map"),
    _randomSeedAction(this, "Random seed"),
    _useClusterColorsAction(this, "Use cluster colors"),
    _embedding(topLevelEmb),
    _embeddingClusters(embeddingClusters)
{
    setText("Mean Shift Top Level");
    setObjectName("Mean Shift Top Level");

    _sigmaAction.setUpdateDuringDrag(false);
    _randomSeedAction.setUpdateDuringDrag(false);
    _numberClustersAction.setEnabled(false);

    _offscreenBuffer = new LocalOffscreenBuffer();

    _offscreenBuffer->bindContext();
    _meanShift.init();
    _offscreenBuffer->releaseContext();

    const auto updateReadOnly = [this]() -> void {
        const auto enabled = !isReadOnly();
        const auto colorBy = static_cast<ColorBy>(_colorByAction.getCurrentIndex());

        _colorMapAction.setEnabled(enabled && colorBy == ColorBy::ColorMap);
        _randomSeedAction.setEnabled(enabled && colorBy == ColorBy::PseudoRandomColors);

    };

    connect(&_colorByAction, &OptionAction::currentIndexChanged, this, [this, updateReadOnly](const std::int32_t& currentIndex) {
        updateColors();
        updateReadOnly();
        });

    connect(&_sigmaAction, &DecimalAction::valueChanged, this, [this](const std::int32_t& currentIndex) {
        compute();
        });

    connect(&_randomSeedAction, &IntegralAction::valueChanged, this, [this](const std::int32_t& currentIndex) {
        updateColors();
        });

    connect(&_colorMapAction, &ColorMapAction::imageChanged, this, [this](const QImage& image) {
        updateColors();
        });

    updateReadOnly();
}

MeanShiftAction::~MeanShiftAction()
{
    delete _offscreenBuffer;
}

void MeanShiftAction::updateColors()
{
    Log::info("MeanShiftAction::updateColors");

    const auto colorBy = static_cast<ColorBy>(_colorByAction.getCurrentIndex());
    auto& clusters = _embeddingClusters->getClusters();

    switch (colorBy) {
    case ColorBy::PseudoRandomColors:
        Cluster::colorizeClusters(clusters, _randomSeedAction.getValue());
        break;

    case ColorBy::ColorMap:
        Cluster::colorizeClusters(clusters, _colorMapAction.getColorMapImage());
        break;
    }

    emit newClusterColors();
    events().notifyDatasetDataChanged(_embeddingClusters);
}

void MeanShiftAction::compute()
{
    Log::info("MeanShiftAction::compute");

    // Remove existing clusters
    _embeddingClusters->getClusters().clear();

    // Update the sigma value
    _meanShift.setSigma(_sigmaAction.getValue());

    std::vector<mv::Vector2f> data;
    _embedding->extractDataForDimensions(data, 0, 1);
    _meanShift.setData(&data);

    std::vector<std::vector<unsigned int>> clusters;

    _offscreenBuffer->bindContext();
    _meanShift.cluster(data, clusters);
    _offscreenBuffer->releaseContext();

    std::int32_t clusterIndex = 0;

    std::vector<std::uint32_t> clusterIDs;
    clusterIDs.resize(_embedding->getNumPoints());

    // Add found clusters
    for (auto& clusterIndicesLocal : clusters)
    {
        Cluster cluster;

        cluster.setName(QString("cluster %1").arg(QString::number(clusterIndex + 1)));

        std::vector<std::uint32_t> clusterIndicesGlobal;

        clusterIndicesGlobal.reserve(clusterIndicesLocal.size());

        for (auto& clusterIndexLocal : clusterIndicesLocal)
        {
            clusterIndicesGlobal.push_back(clusterIndexLocal);
            clusterIDs[clusterIndexLocal] = clusterIndex;

        }

        cluster.setIndices(clusterIndicesGlobal);
        _embeddingClusters->addCluster(cluster);

        clusterIndex++;
    }

    // Update UI
    _numberClustersAction.setValue(static_cast<int32_t>(clusters.size()));

    Log::info(fmt::format("MeanShiftAction:: found {0} clusters using a sigma of {1}", clusters.size(), _sigmaAction.getValue()));

    updateColors(); // emits events().notifyDatasetDataChanged(_embeddingClusters);
}
