#pragma once

#include <QWindow>
#include <QOpenGLContext>
#include <QPointer>

class OffscreenBuffer : public QWindow
{
public:
    OffscreenBuffer();

    QOpenGLContext* getContext() { return _context; }
    bool isInitialized() const { return _isInitialized;  }

    /** Initialize and bind the OpenGL context associated with this buffer */
    void initialize();

    /** Bind the OpenGL context associated with this buffer */
    void bindContext();

    /** Release the OpenGL context associated with this buffer */
    void releaseContext();

private:
    QPointer<QOpenGLContext> _context;
    bool _isInitialized;

};
