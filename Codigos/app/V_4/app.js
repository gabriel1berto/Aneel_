const jsdom = require("jsdom");
const { JSDOM } = jsdom;

const dom = new JSDOM(`<!DOCTYPE html><html><head><title>Dados da ANEEL</title></head><body><table id="data-table"><thead><tr><th>DthAtualizaCadastralEmpreend</th></tr></thead><tbody></tbody></table></body></html>`);
const document = dom.window.document;

fetch('https://dadosabertos.aneel.gov.br/api/action/datastore_search?resource_id=b1bd71e7-d0ad-4214-9053-cbd58e9564a7', { agent: false })
  .then(response => response.json())
  .then(data => {
    const tableBody = document.querySelector('#data-table tbody');
    const columns = Object.keys(data.result.records[0]);
    const headerRow = document.createElement('tr');
    columns.forEach(column => {
      const headerCell = document.createElement('th');
      headerCell.textContent = column;
      headerRow.appendChild(headerCell);
    });
    tableBody.appendChild(headerRow);
    data.result.records.forEach(record => {
      const row = document.createElement('tr');
      columns.forEach(column => {
        const cell = document.createElement('td');
        cell.textContent = record[column];
        row.appendChild(cell);
      });
      tableBody.appendChild(row);
    });
  })
  .catch(error => {
    console.error(error);
  });
