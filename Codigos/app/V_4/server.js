// faz a ponte de conexão com os servidores
const express = require('express');
const port = 3001;
const app = express();

// ativa cors para comunicação
const cors = require('cors');
app.use(cors());

// ativa axios p/ comunicar com API
const axios = require('axios');

let data = {};

// cria rota para buscar informações na API
app.get('/data', async (req, res) => {
    try {
        const response = await axios.get('https://dadosabertos.aneel.gov.br/api/action/datastore_search?resource_id=b1bd71e7-d0ad-4214-9053-cbd58e9564a7');
        data = JSON.parse(JSON.stringify(response.data));
        res.json(data);
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Error fetching data' });
    }
});

// cria o servidor
app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});






// no terminal: C:\Users\ghumb\Desktop\html_php_js\tst.api
// no terminal: node server.js



// no terminal: C:\Users\ghumb\Desktop\html_php_js\tst.api
// no terminal: node server.js