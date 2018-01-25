module.exports = function(grunt) {

	grunt.initConfig({
		comboall: {
			main:{
				files: [
				{'dest/presentation.html': ['src/index.html']}
				]
			}
		}
	});

  grunt.loadNpmTasks('grunt-combo-html-css-js');

  // Default task(s).
  grunt.registerTask('default', ['comboall']);
};
