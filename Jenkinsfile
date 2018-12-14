stage("Build and Publish") {
  node {
    ws('workspace/d2l-zh') {
	  checkout scm
      sh "build/build_all.sh"
      sh """#!/bin/bash
      set -e
      if [[ ${env.BRANCH_NAME} == master ]]; then
          build/upload.sh
      fi
      """
	}
  }
}

