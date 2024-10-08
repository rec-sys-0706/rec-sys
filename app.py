import logging

from server import app

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    app.run(debug=False, host='0.0.0.0', port=5000)
