stage("Build and Publish") {
  node {
    ws('workspace/d2l-zh') {
	  checkout scm
      sh "git submodule update --init --recursive"
      sh "build/utils/clean_build.sh"
      sh "conda env update -f build/env.yml"
      sh "build/utils/build_html.sh zh"
      sh "build/utils/build_pdf.sh zh"
      sh "build/utils/build_pkg.sh zh"
      if (env.BRANCH_NAME == 'master') {
        sh "build/utils/publish_website.sh zh"
      }
    }
  }
}
