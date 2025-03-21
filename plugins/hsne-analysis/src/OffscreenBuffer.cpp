#pragma warning( push ) 
#pragma warning( disable : 4267 ) // disable 'size_t' to 'uint32_t' warning from external library
#include "hdi/dimensionality_reduction/gradient_descent_tsne_texture.h"// Included for glad, must be included before OpenGLContext
#pragma warning( pop ) 

#include "OffscreenBuffer.h"
//#include "Utils.h"

OffscreenBuffer::OffscreenBuffer() :
    _context(new QOpenGLContext(this)),
    _isInitialized(false)
{
    setSurfaceType(QWindow::OpenGLSurface);

    create();
}

void OffscreenBuffer::initialize()
{
//    utils::ScopedTimer InitializeOffscreenBuffer("Initialize GPU offscreen buffer");

    QOpenGLContext* globalContext = QOpenGLContext::globalShareContext();
    _context->setFormat(globalContext->format());

    if (!_context->create())
        qFatal("Cannot create requested OpenGL context.");

    _context->makeCurrent(this);

#ifndef __APPLE__
    if (!gladLoadGL()) {
        qFatal("No OpenGL context is currently bound, therefore OpenGL function loading has failed.");
    }
#endif // Not __APPLE__

    _isInitialized = true;
}

void OffscreenBuffer::bindContext()
{
    _context->makeCurrent(this);
}

void OffscreenBuffer::releaseContext()
{
    _context->doneCurrent();
}
