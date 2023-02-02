#include "neuralnet.h"
#include "qdebug.h"

#include <QPoint>
#include <QRandomGenerator>

// different value functions

namespace
{

using Neuron = NeuralNet::Neuron;
using Connection = Neuron::Connection;
using NeuronIndex = NeuralNet::Index;

// function to generate a double in range [-1,1]
Q_ALWAYS_INLINE qreal generateDouble()
{
    return rng->generateDouble() * 2 - 1;
}

// function to generate random connections to a layer. layerIdx = the index of the "to" layer
inline QList<Connection> connectionsTo(QList<Neuron> to, int layerIdx)
{
    QList<Connection> c;
    const auto len = to.length(); // so we dont need to call the function every time in the loop ("efficiency")
    for(int nI = 0; nI < len; ++nI) {
        c.append(Connection{{layerIdx, nI}, generateDouble()}); // weight should be in the range [-1 : 1]
    }

    return c;
}

// used to normalize the neuron value
Q_ALWAYS_INLINE qreal sigmoid(qreal in)
{
    return 1. / (1 + pow(M_E, -in));
}

// value function, gets the neuron values from the provided layer
Q_ALWAYS_INLINE QList<qreal> grabValues(QList<Neuron> layer)
{
    QList<qreal> d;
    for(const auto &n: layer)
        d.append(n.v);
    return d;
}

qsizetype findConnection(NeuralNet::Index target, QList<Connection> conns)
{
    auto len = conns.length();

    for(qsizetype i = 0; i < len; ++i) {
        auto c = conns[i];
        if(c.target == target)
            return i;
    }

    return -1;
}

// function to crossover 2 neurons
Neuron crossoverNeuron(Neuron n1, Neuron n2, qreal n1Weight = .5, qreal n2Weight = .5)
{
    Neuron child;
    child.bias = n1.bias * n1Weight + n2.bias * n2Weight;
    child.activation = n1.activation * n1Weight + n2.activation * n2Weight;

    // prevent the case of empty connections(invalid neuron) to save runtime
    if(n1.connections.isEmpty()) {
        child.connections = n2.connections;
        return child;
    }

    if(n2.connections.isEmpty()) {
        child.connections = n1.connections;
        return child;
    }

    QList<Connection> c2 = n2.connections;
    QList<Connection> childConns;
    for(const auto &c: n1.connections) {
        auto pIdx = findConnection(c.target, c2);
        if(pIdx == -1) {
            childConns.append(c);
            continue;
        }
        auto pC = c2[pIdx];
        childConns.append(Connection{
                              c.target,
                              c.weight * n1Weight + pC.weight * n2Weight
                          });

        c2.removeAt(pIdx);
    }

    child.connections = childConns;

    return child;
}

} // namespace

// acually the code for the neural net

/***************************************************************************************************
 *                                        INFO ABOUT THE NET                                       *
 ***************************************************************************************************/
/*
 * the net looks something like this:
 *
 *      input
 * hidden layer 1
 * hidden layer 2
 * hidden layer 3 ...
 *      output
 */

NeuralNet::NeuralNet(int input, int hiddenLayers, int hLS, int output, int maxLayerWidth)
    : m_neurons({QList<Neuron>(input)}) // pre-generate the input neurons, since they need noting special
    , m_maxLayerW(maxLayerWidth)
{
    QList<Neuron> currentLayer;

    // generate the hidden layers
    for(int hL = 0; hL < hiddenLayers; ++hL) {
        auto parentLayer = m_neurons[hL];
        for(int hN = 0; hN < hLS; ++hN) {
            currentLayer.append(Neuron(connectionsTo(parentLayer, hL)));
        }

        m_neurons.append(currentLayer);
        currentLayer.clear();
    }

    // generate the output neurons
    auto lastHiddenLayer = m_neurons.last();
    auto lastHiddenIndex = m_neurons.length() - 1;
    for(int nI = 0; nI < output; ++nI) {
        currentLayer.append(Neuron(connectionsTo(lastHiddenLayer, lastHiddenIndex)));
    }
    m_neurons.append(currentLayer);
}

void NeuralNet::crossover(NeuralNet *partner)
{
    auto layerAmt = qMax(m_neurons.length(), partner->neurons().length());
    QList<QList<Neuron>> childNeurons;

    for(int l = 0; l < layerAmt; ++l) {
        // catch the case of a net having less layers than the other
        if(l >= m_neurons.length() - 1 && !(l == layerAmt - 1)) { // output layer and it is not the last iteration
            childNeurons.append(partner->neurons()[l]);
            continue;
        }
        if(l >= partner->neurons().length() - 1 && !(l == layerAmt - 1)) { // same for the partner net
            childNeurons.append(m_neurons[l]);
            continue;
        }

        const auto myLayer = m_neurons[l];
        const auto partnerLayer = partner->neurons()[l];
        QList<Neuron> childLayer;

        auto neuronAmt = qMax(myLayer.length(), partnerLayer.length());

        // crossover the layers
        for(int n = 0; n < neuronAmt; ++n) {
            if(n >= myLayer.length() - 1 && !(n == neuronAmt - 1)) { // last layer and it is not the last iteration
                childLayer.append(partnerLayer[n]);
                continue;
            }
            if(n >= partnerLayer.length() - 1 && !(n == neuronAmt - 1)) { // same for the partner layer
                childLayer.append(myLayer[n]);
                continue;
            }

            childLayer.append(crossoverNeuron(myLayer[n], partnerLayer[n]));
        }

        childNeurons.append(childLayer);
    }

    m_neurons = childNeurons;
}

NeuralNet *NeuralNet::crossover(NeuralNet *a, NeuralNet *b)
{
    auto layerAmt = qMax(a->neurons().length(), b->neurons().length());
    QList<QList<Neuron>> childNeurons;

    for(int l = 0; l < layerAmt; ++l) {
        // catch the case of a net having less layers than the other
        if(l >= a->neurons().length() - 1 && !(l == layerAmt - 1)) { // output layer and it is not the last iteration
            childNeurons.append(b->neurons()[l]);
            continue;
        }
        if(l >= b->neurons().length() - 1 && !(l == layerAmt - 1)) { // same for the b net
            childNeurons.append(a->neurons()[l]);
            continue;
        }

        const auto aLayer = a->neurons()[l];
        const auto bLayer = b->neurons()[l];
        QList<Neuron> childLayer;

        auto neuronAmt = qMax(aLayer.length(), bLayer.length());

        // crossover the layers
        for(int n = 0; n < neuronAmt; ++n) {
            if(n >= aLayer.length()) { // last layer and it is not the last iteration
                childLayer.append(bLayer[n]);
                continue;
            }
            if(n >= bLayer.length()) { // same for the b layer
                childLayer.append(aLayer[n]);
                continue;
            }

            childLayer.append(crossoverNeuron(aLayer[n], bLayer[n]));
        }

        childNeurons.append(childLayer);
    }

    return new NeuralNet(childNeurons);
}

QList<qreal> NeuralNet::decide(QList<qreal> in)
{
#ifdef NEURALNET_DURATION_INFO
    static auto clock = std::chrono::high_resolution_clock();
    auto start = clock.now();
#endif // NEURALNET_DURATION_INFO

    // set the input:
    auto &inputLayer = m_neurons[0];
    auto inputLen = inputLayer.length();
    for(int inIdx = 0; inIdx < inputLen; ++inIdx) {
        inputLayer[inIdx].v = in[inIdx];
    }

    const auto layerAmt = m_neurons.length();
    for(int layerIdx = 1; layerIdx < layerAmt; ++layerIdx) {
        const auto neuronAmt = m_neurons[layerIdx].length();
        for(int neuronIdx = 0; neuronIdx < neuronAmt; ++neuronIdx) {

            // decide for every neuron
            auto &n = m_neurons[layerIdx][neuronIdx];

            qreal v = 0;

            for(auto c: n.connections) {
                v += neuronAt(c.target).v * c.weight;
            }

            // normalize the value
            v /= n.bias;
            v = sigmoid(v) * 2 - 1;

            // if its less than the activation, the value gets set to 0
            if(v > n.activation)
                n.v = v;
            else
                n.v = 0;
        }
    }

    auto out = grabValues(m_neurons.last());

#ifdef NEURALNET_DURATION_INFO
    auto end = clock.now();
    std::chrono::nanoseconds _diff = (end - start);
    qreal diff = _diff.count() * 1. / (std::nano::den / std::milli::den); // nano -> ms

    qInfo().nospace() << Q_FUNC_INFO << ": deciding took " << diff << "ms";
#endif // NEURALNET_DURATION_INFO
    return out;
}

void NeuralNet::mutate(qreal mr)
{
    // mutating works the following way:
    /*
     * run through every layer, and then run through every neuron in the current layer. do the following instructions for every neuron:
     *  1. generate a random number in range [1,0] (use QRandomGenerator::generateDouble)
     *  2. if it is smaller than mr, mutate the neuron.
     *   for that, change the activation and bias by a random value in range [-1,1]. if any of the values goes out of range [-1,1], crop it.
     *   run through every connection and do the same thing as above for the weight.
     */

#ifdef NEURALNET_DURATION_INFO
    static auto clock = std::chrono::high_resolution_clock();
    auto start = clock.now();
#endif // NEURALNET_DURATION_INFO

    const int layerAmt = m_neurons.length();
    for(int lI = 1; lI < layerAmt; ++lI) { // skip the first layer since the neurons don't do anything so don't need to be mutated
        auto &layer = m_neurons[lI];

        for(auto &n: layer) {
            // check if we mutate the current neuron
            if(!(rng->generateDouble() < mr)) // its not smaller -> continue with the nex neuron
                continue;

            // start by mutating the bias and activation
            n.bias += generateDouble();
            n.activation += generateDouble();

            // crop the values
            if(n.bias < -1) n.bias = -1;
            if(n.bias > 1)  n.bias = 1;
            if(n.activation < -1) n.activation = -1;
            if(n.activation > 1)  n.activation = 1;

            // mutate the connections
            for(auto &c: n.connections) {
                c.weight += generateDouble();

                // als crop the weight
                if(c.weight < -1) c.weight = -1;
                if(c.weight > 1)  c.weight = 1;
            }
        }
    }

    // add a neuron with a 10% rate
    if(rng->generateDouble() < .1)
        addNeuron();

    // remove a neuron with a 10% rate
    if(rng->generateDouble() < .1)
        removeNeuron();

#ifdef NEURALNET_DURATION_INFO
    auto end = clock.now();
    std::chrono::nanoseconds _diff = (end - start);
    qreal diff = _diff.count() * 1. / (std::nano::den / std::milli::den); // nano -> ms

    qInfo().nospace() << Q_FUNC_INFO << ": mutation took " << diff << "ms";
#endif // NEURALNET_DURATION_INFO
}

void NeuralNet::addNeuron()
{
    // adding a neuron works the following way: we add a neuron to a random layer and then add connections from the next layer to it
    const int layerIdx = rng->bounded(1, m_neurons.length() - 1); // exclude input and output layer
    auto &layer = m_neurons[layerIdx];

    const int neuronIdx = layer.length();
    if(neuronIdx > m_maxLayerW) {
        qInfo() << "failed to add neuron to layer" << layerIdx << ": neuron index" << neuronIdx << "is out of range.";
        return;
    }

    Neuron newN(connectionsTo(m_neurons[layerIdx - 1], layerIdx - 1));


    auto &nextLayer = m_neurons[layerIdx + 1];

    for(auto &n: nextLayer)
        n.connections.append(Connection{{layerIdx, neuronIdx}, generateDouble()});

    layer.append(newN);

    qInfo() << Q_FUNC_INFO << layerIdx << neuronIdx;
}

void NeuralNet::removeNeuron()
{
    // removing a neuron works the following way: we remove a neuron with a random index from a random layer. then we remove the connections to it
    const int lI = rng->bounded(1, m_neurons.length() - 1);
    if(m_neurons[lI].length() == 1) {
        return;
    }
    const int nI = rng->bounded(0, m_neurons[lI].length());

    qInfo() << "starting to remove neuron" << lI << nI;

    // remove the choosen neuron
    m_neurons[lI].removeAt(nI);

    // remove the connections to the neuron. if there is a connection to it, delete it; if the target neuron index is greater than the index oft the choosen neuron lower it by 1
    auto &nextLayer = m_neurons[lI + 1];
    for(auto &n: nextLayer) {
        int cA = n.connections.length();
        for(int cI = 0; cI < cA; ++cI) {
            auto &c = n.connections[cI];
            if(c.target.neuron == nI) {
                n.connections.removeAt(cI);
                // we need to lower the amount of neurons since we removed one.
                --cA;
                --cI;
            }
            if(c.target.neuron > nI) {
//                qInfo() << "pre:" << toPoint(c.target);
                auto &t = c.target;
                t.neuron -= 1;
//                qInfo() << "after:" << toPoint(c.target);
            }
        }
    }
//    qInfo() << "no i dont kill my neurons :( LMAAOAAOAAO";
}

inline NeuralNet::Neuron NeuralNet::neuronAt(Index idx)
{
    Q_ASSERT_X(idx.isValid(), Q_FUNC_INFO, ("invalid index " + QString::number(idx.layer) + " | " + QString::number(idx.neuron)).toUtf8());
    return m_neurons[idx.layer][idx.neuron];
}

void NeuralNet::set(QList<QList<Neuron> > d)
{
//    qInfo() << m_neurons.length();
//    if(!m_neurons.isEmpty())
//        m_neurons.clear();
    m_neurons.swap(d);
}

// conversion functions
QPoint toPoint(NeuralNet::Index idx)
{
    Q_ASSERT_X(idx.isValid(), Q_FUNC_INFO, ("invalid index " + QString::number(idx.layer) + " | " + QString::number(idx.neuron)).toUtf8());
    return QPoint(idx.layer, idx.neuron);
}

NeuralNet::Index toIndex(QPoint p)
{
    Q_ASSERT_X(p.x() >= 0 && p.y() >= 0, Q_FUNC_INFO, ("invalid point QPoint(" + QString::number(p.x()) + ", " + QString::number(p.y()) + ")").toUtf8());

    return NeuralNet::Index { p.x(), p.y() };
}

// reading and writing functions
    // NeuralNet write
void writeNeuralNet(QDataStream &s, NeuralNet *net)
{
    s << net->m_maxLayerW << net->neurons().length();
    for(const auto &l: net->neurons()) {
        s << l.length();
        for(const auto n: l) {
            s << n;
        }
    }
}
QDataStream &operator<<(QDataStream &s, NeuralNet *net)
{ writeNeuralNet(s, net); return s; }

    // NeuralNet read
void readNeuralNet(QDataStream &s, NeuralNet *net)
{
    if(net)
        delete net; // delete the old net

    int maxLayerW;
    qsizetype layerAmt;
    QList<QList<NeuralNet::Neuron>> layers;
    s >> maxLayerW >> layerAmt;

    qInfo() << "-- net has" << layerAmt << "layers";

    for(int l = 0; l < layerAmt; ++l) {
        qsizetype layerW;
        s >> layerW;

        qInfo() << "-- -- layer" << l << "has" << layerW << "neurons";

        QList<NeuralNet::Neuron> layer;

        for(int nI = 0; nI < layerW; ++nI) {
            NeuralNet::Neuron n;
            s >> n;
            layer.append(n);
        }

        layers.append(layer);
    }

    net = new NeuralNet(layers);
}

QDataStream &operator>>(QDataStream &s, NeuralNet *net)
{ readNeuralNet(s, net); return s; }

    // Neuron write
void writeNeuron(QDataStream &s, NeuralNet::Neuron neuron)
{
    s << neuron.bias << neuron.activation << neuron.connections.length();
    for(auto &c: neuron.connections) {
        s << c.weight << toPoint(c.target);
    }
}
QDataStream &operator<<(QDataStream &s, NeuralNet::Neuron neuron)
{ writeNeuron(s, neuron); return s; }

    // Neuron read
void readNeuron(QDataStream &s, Neuron &n)
{
    qreal bias;
    qreal activation;
    qsizetype connectionAmt;
    QList<NeuralNet::Neuron::Connection> connections;

    s >> bias >> activation >> connectionAmt;
    for(int i = 0; i < connectionAmt; ++i) {
        qreal w;
        QPoint target;
        NeuralNet::Neuron::Connection c;
        s >> w >> target;

        c = {toIndex(target), w};
        connections.append(c);
    }

    n = {connections, bias, activation};
}
QDataStream &operator>>(QDataStream &s, Neuron &n)
{ readNeuron(s, n); return s; }
