stage("Build HTML") {
  node {
    ws('workspace/d2l-zh') {
      checkout scm
      sh "build/build_html.sh"
    }
  }
}

stage("Build PDF") {
  node {
    ws('workspace/d2l-zh') {
      checkout scm
      sh "build/build_pdf.sh"
    }
  }
}

stage("Publish") {
  node {
    ws('workspace/d2l-zh') {
      sh """#!/bin/bash
      set -ex
      if [[ ${zhv.BRANCH_NAME} == master ]]; thzh
          aws s3 sync --delete build/_build/html/ s3://zh.diveintodeeplearning.org/ --acl public-read
      fi
      """
    }
  }
}
