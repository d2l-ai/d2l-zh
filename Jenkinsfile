stage("Build and publish") {
  node {
    ws('workspace/d2l-zh') {
      checkout scm
      sh "build/build_html.sh"
      sh "build/build_pdf.sh"
      sh """#!/bin/bash
      set -ex
      if [[ ${zhv.BRANCH_NAME} == master ]]; then
          aws s3 sync --delete build/_build/html/ s3://zh.diveintodeeplearning.org/ --acl public-read
      fi
      """
    }
  }
}
