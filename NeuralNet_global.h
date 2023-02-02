#ifndef NEURALNET_GLOBAL_H
#define NEURALNET_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(NEURALNET_LIBRARY)
#  define NEURALNET_EXPORT Q_DECL_EXPORT
#else
#  define NEURALNET_EXPORT Q_DECL_IMPORT
#endif

#ifndef INFO
 #define INFO(d) #d << ":" << d
 #define FUNC_INFO qInfo() << Q_FUNC_INFO << ":"
#endif

#ifndef rng
 #define rng QRandomGenerator::global()
#endif // rng

#if defined(QT_DEBUG) && 0
#define NEURALNET_DURATION_INFO
#endif

#endif // NEURALNET_GLOBAL_H
