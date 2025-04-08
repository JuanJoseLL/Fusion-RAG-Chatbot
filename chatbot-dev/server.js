const express = require('express');
const path = require('path');
const app = express();
const PORT = 3000;
const axios = require('axios');


app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());



app.post('/chat', async (req, res) => {
  const { message } = req.body;

  try {
    const response = await axios.post('http://localhost:8000/chat', new URLSearchParams({ message }), {
     
    });

    res.json({ reply: response.data.reply });
  } catch (err) {
    console.error(err);
    res.status(500).json({ reply: "Error al comunicarse con el backend." });
  }
});


app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'test-client.html'));
});

app.listen(PORT, () => {
  console.log(`Servidor corriendo en http://localhost:${PORT}`);
});
