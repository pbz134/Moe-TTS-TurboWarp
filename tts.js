(function(Scratch) {
  'use strict';

  class TTSExtension {
    getInfo() {
      return {
        id: 'ttsExtension',
        name: 'Text To Speech',
        blocks: [
          {
            opcode: 'setSpeaker',
            blockType: Scratch.BlockType.COMMAND,
            text: 'set speaker to [SPEAKER]',
            arguments: {
              SPEAKER: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 0
              }
            }
          },
          {
            opcode: 'setText',
            blockType: Scratch.BlockType.COMMAND,
            text: 'set text to [TEXT]',
            arguments: {
              TEXT: {
                type: Scratch.ArgumentType.STRING,
                defaultValue: 'こんにちは'
              }
            }
          },
          {
            opcode: 'setSpeed',
            blockType: Scratch.BlockType.COMMAND,
            text: 'set speed to [SPEED]',
            arguments: {
              SPEED: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 1
              }
            }
          },
          {
            opcode: 'setModel',
            blockType: Scratch.BlockType.COMMAND,
            text: 'set model to [MODEL]',
            arguments: {
              MODEL: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 0
              }
            }
          },
          {
            opcode: 'setVolume',
            blockType: Scratch.BlockType.COMMAND,
            text: 'set volume to [VOLUME]',
            arguments: {
              VOLUME: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 1
              }
            }
          },
          {
            opcode: 'setPan',
            blockType: Scratch.BlockType.COMMAND,
            text: 'set pan to [PAN]',
            arguments: {
              PAN: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 0
              }
            }
          },
          {
            opcode: 'speak',
            blockType: Scratch.BlockType.COMMAND,
            text: 'speak',
          },
          {
            opcode: 'testConnection',
            blockType: Scratch.BlockType.REPORTER,
            text: 'test connection',
          },
          {
            opcode: 'stopSpeaking',
            blockType: Scratch.BlockType.COMMAND,
            text: 'stop speaking',
          }
        ]
      };
    }

    constructor() {
      this.speaker = 0;
      this.text = 'こんにちは';
      this.speed = 1;
      this.model = 0;
      this.volume = 1;
      this.pan = 0;
      this.audio = null;
    }

    setSpeaker(args) {
      this.speaker = args.SPEAKER;
    }

    setText(args) {
      this.text = args.TEXT;
    }

    setSpeed(args) {
      this.speed = args.SPEED;
    }

    setModel(args) {
      this.model = args.MODEL;
    }

    setVolume(args) {
      this.volume = args.VOLUME;
    }

    setPan(args) {
      this.pan = args.PAN;
    }

    speak() {
      const text = this.text;
      const speaker = this.speaker;
      const speed = this.speed;
      const model = this.model;

      const apiUrl = 'http://127.0.0.1:5000/tts'; // Flask API URL

      fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text, speaker, speed, model_id: model })
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.blob();
      })
      .then(blob => {
        const url = URL.createObjectURL(blob);
        this.audio = new Audio(url);
        this.audio.volume = this.volume;
        this.audio.pan = this.pan;
        this.audio.play();
      })
      .catch(error => {
        console.error('Error:', error);
      });
    }

    stopSpeaking() {
      if (this.audio) {
        this.audio.pause();
        this.audio.currentTime = 0;
      }
    }

    testConnection() {
      const apiUrl = 'http://127.0.0.1:5000/tts'; // Flask API URL

      return fetch(apiUrl, {
        method: 'OPTIONS',
        headers: {
          'Content-Type': 'application/json'
        }
      })
      .then(response => {
        if (response.ok) {
          return 'Connection successful';
        } else {
          return `Connection failed with status: ${response.status}`;
        }
      })
      .catch(error => {
        return `Connection error: ${error.message}`;
      });
    }
  }

  Scratch.extensions.register(new TTSExtension());
})(Scratch);
