#include "Utils.h"

#include "hdi/dimensionality_reduction/knn_utils.h"

namespace utils {

    /// ////////// ///
    /// EMBEDDINGS ///
    /// ////////// ///

    EmbeddingExtends::EmbeddingExtends() : _x_min(0), _x_max(0), _y_min(0), _y_max(0), _extend_x(0), _extend_y(0) {}
    EmbeddingExtends::EmbeddingExtends(float x_min, float x_max, float y_min, float y_max) : _x_min(x_min), _x_max(x_max), _y_min(y_min), _y_max(y_max)
    {
        _extend_x = _x_max - _x_min;
        _extend_y = _y_max - _y_min;
    }

    void EmbeddingExtends::setExtends(float x_min, float x_max, float y_min, float y_max) {
        if (!(x_min < x_max))
            Log::warn("EmbeddingExtends::setExtends: x_min < x_max");
        if (!(y_min < y_max))
            Log::warn("EmbeddingExtends::setExtends: y_min < y_max");

        _x_min = x_min; _x_max = x_max;
        _y_min = y_min; _y_max = y_max;

        _extend_x = _x_max - _x_min;
        _extend_y = _y_max - _y_min;
    }

    EmbeddingExtends computeExtends(const std::vector<float>& emb)
    {
        float x_min(0), x_max(0), y_min(0), y_max(0);

        auto data = emb.data();
        for (size_t i = 0; i < emb.size() / 2; i++)
        {
            if (x_min > data[i * 2]) x_min = data[i * 2];
            if (x_max < data[i * 2]) x_max = data[i * 2];
            if (y_min > data[i * 2 + 1]) y_min = data[i * 2 + 1];
            if (y_max < data[i * 2 + 1]) y_max = data[i * 2 + 1];
        }

        return { x_min, x_max, y_min, y_max };
    }

    EmbeddingExtends computeExtends(const std::vector<mv::Vector2f>& emb)
    {
        float x_min(0), x_max(0), y_min(0), y_max(0);

        auto data = emb.data();
        for (size_t i = 0; i < emb.size() / 2; i++)
        {
            if (x_min > data[i].x) x_min = data[i].x;
            if (x_max < data[i].x) x_max = data[i].x;
            if (y_min > data[i].y) y_min = data[i].y;
            if (y_max < data[i].y) y_max = data[i].y;
        }

        return { x_min, x_max, y_min, y_max };
    }


    /// ////// ///
    /// TIMING ///
    /// ////// ///

    ScopedTimer::ScopedTimer(std::string title, std::function<void(std::string)> _logFunc) : _start(clock::now()), _title(title), _logFunc(_logFunc) {}

    ScopedTimer::~ScopedTimer() {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - _start).count();
        _logFunc(fmt::format("Duration of {0}: {1} ms", _title, duration));
    }

    /// ///// ///
    /// ENUMS ///
    /// ///// ///

    bool convertToHDILibKnnLib(const utils::knn_library& in, hdi::dr::knn_library& out)
    {
        const auto hdi_knn_libs = hdi::dr::supported_knn_libraries();

        if (static_cast<unsigned int>(in) >= hdi_knn_libs.size())
            return false;

        out = static_cast<hdi::dr::knn_library>(in);
        return true;
    }


}