
let modelo;

async function run() {
    const data = await getData();
    vizualizarDatos(data);
    modelo = crearModelo();

    const tensorData = convertirDatosATensores(data);
    const { entradas, etiquetas } = tensorData;

    entrendarModelo(modelo, entradas, etiquetas);
}

async function entrendarModelo(modelo, entradas, etiquetas) {
    modelo.compile({
        /* optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'], */
        optimizer: optimizador,
        loss: perdida,
        metrics: metricas,
    });

    const surface = { name: 'Muestra historial ', tab: 'Training' };
    const tamanioBatch = 28;
    const epochs = 50;
    const history = [];
    return modelo.fit(entradas, etiquetas, {
        tamanioBatch,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            surface,
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}


const optimizador = tf.train.adam();
const perdida = tf.losses.meanSquaredError;
const metricas = ['mse'];

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
    });


    return {
        entradas: entradasNormalizadas,
        etiquetas: etiquetasNormalizadas,
        entradasMax,
        entradasMin,
        etiquetasMax,
        etiquetasMin
    }

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
            }
        );
    }

    function crearModelo() {
        const modelo = tf.sequential();

        modelo.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
        modelo.add(tf.layers.dense({ units: 1, useBias: true }));

        return modelo;
    }


    
    async function getData() {
        const datosCasasR = await fetch('datos.json');
        const datoCasas = await datosCasasR.json();
        var datosLimpios = datoCasas.map(casa => {

            ({
                precio: casa.Precio,
                cuartos: casa.NumeroDeCuartosPromedio
            })
        });

        datosLimpios = datoCasas.filter(casa => {
            casa.Precio != null && casa.NumeroDeCuartosPromedio != null
            return datosLimpios;
        });

    }

    