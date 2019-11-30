'use strict';

const path = require('path');
const { readdirSync, statSync } = require('fs');

const CopyWebpackPlugin = require('copy-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const ImageminPlugin = require('imagemin-webpack-plugin').default;
const ImageminWebpWebpackPlugin = require('imagemin-webp-webpack-plugin');
const SizePlugin = require('size-plugin');
const WriteFilePlugin = require('write-file-webpack-plugin');

module.exports = (env, argv) => ({
	devtool: 'sourcemap',
	devServer: {
		contentBase: [
			'./dist',
			'./'
		],
		hot: true,
		writeToDisk: true,
		watchContentBase: true
	},
	stats: 'errors-only',
	entry: {
		'index': './src/js/index'
	},
	output: {
		path: path.join(__dirname, 'dist'),
		filename: 'js/[name].js',
		publicPath: '',
	},
	module: {
		rules: [{
			test: /\.pug/,
			use: 'pug-loader'
		}]
	},
	plugins: [
		new SizePlugin(),
		new WriteFilePlugin(),
		new CopyWebpackPlugin([
			{
				from: 'assets',
				to: 'assets'
			}
		]),
		new HtmlWebpackPlugin({
			template: './src/pug/index.pug',
			filename: 'index.html',
			inject: false
		}),
		new ImageminPlugin(),
		new ImageminWebpWebpackPlugin({
			config:[
				{
					test: /assets\/img/
				}
			],
			detailedLogs: true
		})
	],
	resolve: {
		extensions: [
			'.js'
		]
	},
	optimization: {
		concatenateModules: true,
	}
});
