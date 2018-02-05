var express = require('express');
var app = express();

var redis = require('redis');
var client = redis.createClient('//lab4.hwanmoo.kr:6379/0');

var http = require('http');
var fs = require('fs');

// var file = fs.createWriteStream("file.mp4");
// var request = http.get("http://localhost:5000/device1.mp4", function(response) {
//   response.pipe(file);
// });

client.on("message", function(channel, message) {
  console.log("Message '" + message + "' on channel '" + channel + "' arrived!")
});

client.subscribe("message");

app.get('/', function (req, res) {
  res.send('Hello World!');
});

app.listen(3000, function () {
  console.log('Example app listening on port 3000!');
});
