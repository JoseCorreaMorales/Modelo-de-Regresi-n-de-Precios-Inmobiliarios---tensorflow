let modelo;
let stopTraining = false;

async function getData() {
    const datosCasasR = await fetch("datos.json");
    const datoCasas = await datosCasasR.json();
    var datosLimpios = datoCasas.map(casa => (
        {
            precio: casa.Precio,
            cuartos: casa.NumeroDeCuartosPromedio
        }
    ));

    datosLimpios = datosLimpios.filter(casa => (
        casa.precio != null && casa.cuartos != null  
    ));
    return datosLimpios;
}

function vizualizarDatos(data) {
    const valores = data.map(d => ({ x: d.cuartos, y: d.precio }));

    tfvis.render.scatterplot(
        { name: 'Numero de cuartos vs Precio' },
        { values: valores },
        {
            xLabel: 'Numero de cuartos',
            yLabel: 'Precio',
            height: 300
        });
}

function crearModelo() {
    const modelo = tf.sequential();
    modelo.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
    modelo.add(tf.layers.dense({ units: 1, useBias: true }));
    return modelo;
}
function convertirDatosATensores(data) {
    return tf.tidy(() => {
        tf.util.shuffle(data);
        const entradas = data.map(d => d.cuartos);
        const etiquetas = data.map(d => d.precio);

        const tensorEntradas = tf.tensor2d(entradas, [entradas.length, 1]);
        const tensorEtiquetas = tf.tensor2d(etiquetas, [etiquetas.length, 1]);

        const entradasMax = tensorEntradas.max();
        const entradasMin = tensorEntradas.min();
        const etiquetasMax = tensorEtiquetas.max();
        const etiquetasMin = tensorEtiquetas.min();

        const entradasNormalizadas = tensorEntradas.sub(entradasMin).div(entradasMax.sub(entradasMin));
        const etiquetasNormalizadas = tensorEtiquetas.sub(etiquetasMin).div(etiquetasMax.sub(etiquetasMin));

        return {
            entradas: entradasNormalizadas,
            etiquetas: etiquetasNormalizadas,
            entradasMax,
            entradasMin,
            etiquetasMax,
            etiquetasMin
        }
    });}

const optimizador = tf.train.adam();
const perdida = 'meanSquaredError';
const metricas = ['mse'];

async function entrenarModelo(modelo, entradas, etiquetas) {
    modelo.compile({
        optimizer: optimizador,
        loss: perdida,
        metrics: metricas,
    });

    const surface = { name: 'Muestra historial', tab: 'Training' };
    const tamanioBatch = 28;
    const epochs = 50;
    const history = [];
    
    return modelo.fit(entradas, etiquetas, {
        batchSize: tamanioBatch,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'mse'], { callbacks: ['onEpochEnd'] }),
    });
}

async function cargarModelo() {
    const uploadJSONInput = document.getElementById('upload-json');
    const uploadWeightsInput = document.getElementById('upload-weights');

    modelo = await tf.loadLayersModel(tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
    console.log('Modelo cargado desde disco');
}

async function guardarModelo() {
    const saveResult = await modelo.save('downloads://modelo_regresion');
}

async function verCurvaInferencia() {
    const data = await getData();
    const tensorData = convertirDatosATensores(data);
    const { entradasMax, entradasMin, etiquetasMin, etiquetasMax } = tensorData;

    const xs = tf.linspace(0, 1, 100);
    const preds = modelo.predict(xs.reshape([100, 1]));

    const desnormX = xs.mul(entradasMax.sub(entradasMin)).add(entradasMin);
    const desnormY = preds.mul(etiquetasMax.sub(etiquetasMin)).add(etiquetasMin);

    const puntosPrediccion = Array.from(desnormX.dataSync()).map((val, i) => {
        return { x: val, y: desnormY.dataSync()[i] }
    });

    const puntosOriginales = data.map(d => ({
        x: d.cuartos, y: d.precio,
    }));

    tfvis.render.scatterplot(
        { name: 'Predicciones vs Originales' },
        { values: [puntosOriginales, puntosPrediccion], series: ['original', 'prediccion'] },
        {
            xLabel: 'Numero de cuartos',
            yLabel: 'Precio',
            height: 300
        }
    );
}

async function run() {
    const data = await getData();
    vizualizarDatos(data);
    modelo = crearModelo();

    const tensorData = convertirDatosATensores(data);
    const { entradas, etiquetas } = tensorData;

    await entrenarModelo(modelo, entradas, etiquetas);
}

run();
