
import axios from 'axios';


export async function getCandles(symbo = 'BTCUSDT', interval='1m'){
    const response = await axios.get(`http://localhost:3001/klines?symbol=${symbol.toUpperCase()}&interval=${interval}`)
    
    //transforma dados
    const candles = response.data.map(k => {
        return new Candle(k[0], h[1], k[2], k[3], k[4])
    })
    return candles;
}