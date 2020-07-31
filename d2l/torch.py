




<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
  <link rel="dns-prefetch" href="https://github.githubassets.com">
  <link rel="dns-prefetch" href="https://avatars0.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars1.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars2.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars3.githubusercontent.com">
  <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com">
  <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">



  <link crossorigin="anonymous" media="all" integrity="sha512-YR0i2ZAJ3fFf7L2CvMny+FWH76iHZNNIcD1YX57o4cdBHev8ffMXOfzy5F/lpBJpLttwPahk3zY/8XXaRH12ew==" rel="stylesheet" href="https://github.githubassets.com/assets/frameworks-611d22d99009ddf15fecbd82bcc9f2f8.css" />
  <link crossorigin="anonymous" media="all" integrity="sha512-fCe2u2fIROPr55bT48P5R9OCi4OYOnm5P30FSdeqB3c0TqaPIe0x3d6smDmrOQ9ubiAvbWmFeXUb5rTzRhZf8w==" rel="stylesheet" href="https://github.githubassets.com/assets/site-7c27b6bb67c844e3ebe796d3e3c3f947.css" />
    <link crossorigin="anonymous" media="all" integrity="sha512-WQpDCIwNh3ihS4etXWMXxcG0IJ0g2Hw3uFb1nDZxWJCw97dFqda3zkqBOXr5tb/O5V5gAxh1nxN1fIeNJ6GzHw==" rel="stylesheet" href="https://github.githubassets.com/assets/github-590a43088c0d8778a14b87ad5d6317c5.css" />
    
    
    
    


  <meta name="viewport" content="width=device-width">
  
  <title>d2l-en/torch.py at master · d2l-ai/d2l-en · GitHub</title>
    <meta name="description" content="Interactive deep learning book with code, math, and discussions. Available in multi-frameworks. - d2l-ai/d2l-en">
    <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub">
  <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
  <meta property="fb:app_id" content="1401488693436528">
  <meta name="apple-itunes-app" content="app-id=1477376905">

    <meta name="twitter:image:src" content="https://avatars3.githubusercontent.com/u/43974506?s=400&amp;v=4" /><meta name="twitter:site" content="@github" /><meta name="twitter:card" content="summary" /><meta name="twitter:title" content="d2l-ai/d2l-en" /><meta name="twitter:description" content="Interactive deep learning book with code, math, and discussions. Available in multi-frameworks. - d2l-ai/d2l-en" />
    <meta property="og:image" content="https://avatars3.githubusercontent.com/u/43974506?s=400&amp;v=4" /><meta property="og:site_name" content="GitHub" /><meta property="og:type" content="object" /><meta property="og:title" content="d2l-ai/d2l-en" /><meta property="og:url" content="https://github.com/d2l-ai/d2l-en" /><meta property="og:description" content="Interactive deep learning book with code, math, and discussions. Available in multi-frameworks. - d2l-ai/d2l-en" />

  <link rel="assets" href="https://github.githubassets.com/">
  

  <meta name="request-id" content="CFD8:143D:4675F:62E35:5F246763" data-pjax-transient="true"/><meta name="html-safe-nonce" content="88a4c1614d5e2196354db51972d0ceaeeb6ebb3c" data-pjax-transient="true"/><meta name="visitor-payload" content="eyJyZWZlcnJlciI6IiIsInJlcXVlc3RfaWQiOiJDRkQ4OjE0M0Q6NDY3NUY6NjJFMzU6NUYyNDY3NjMiLCJ2aXNpdG9yX2lkIjoiNDcxMDcyODkwNzk0ODI5NjY3IiwicmVnaW9uX2VkZ2UiOiJzZWEiLCJyZWdpb25fcmVuZGVyIjoic2VhIn0=" data-pjax-transient="true"/><meta name="visitor-hmac" content="62770537cb6445cd67f875dbe8f4c55f20bdd02526d85918239f71e18695e271" data-pjax-transient="true"/>

    <meta name="hovercard-subject-tag" content="repository:152166877" data-pjax-transient>


  <meta name="github-keyboard-shortcuts" content="repository,source-code" data-pjax-transient="true" />

  

  <meta name="selected-link" value="repo_source" data-pjax-transient>

    <meta name="google-site-verification" content="c1kuD-K2HIVF635lypcsWPoD4kilo5-jA_wBFyT4uMY">
  <meta name="google-site-verification" content="KT5gs8h0wvaagLKAVWq8bbeNwnZZK1r1XQysX3xurLU">
  <meta name="google-site-verification" content="ZzhVyEFwb7w3e0-uOTltm8Jsck2F5StVihD0exw2fsA">
  <meta name="google-site-verification" content="GXs5KoUUkNCoaAZn7wPN-t01Pywp9M3sEjnt_3_ZWPc">

  <meta name="octolytics-host" content="collector.githubapp.com" /><meta name="octolytics-app-id" content="github" /><meta name="octolytics-event-url" content="https://collector.githubapp.com/github-external/browser_event" /><meta name="octolytics-dimension-ga_id" content="" class="js-octo-ga-id" />

  <meta name="analytics-location" content="/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show" data-pjax-transient="true" />

  




    <meta name="google-analytics" content="UA-3769691-2">


<meta class="js-ga-set" name="dimension10" content="Responsive" data-pjax-transient>

<meta class="js-ga-set" name="dimension1" content="Logged Out">



  

      <meta name="hostname" content="github.com">
    <meta name="user-login" content="">


      <meta name="expected-hostname" content="github.com">


    <meta name="enabled-features" content="MARKETPLACE_PENDING_INSTALLATIONS">

  <meta http-equiv="x-pjax-version" content="09cfdfcc0ba267bba72b85c0824e558f">
  

      <link href="https://github.com/d2l-ai/d2l-en/commits/master.atom" rel="alternate" title="Recent Commits to d2l-en:master" type="application/atom+xml">

  <meta name="go-import" content="github.com/d2l-ai/d2l-en git https://github.com/d2l-ai/d2l-en.git">

  <meta name="octolytics-dimension-user_id" content="43974506" /><meta name="octolytics-dimension-user_login" content="d2l-ai" /><meta name="octolytics-dimension-repository_id" content="152166877" /><meta name="octolytics-dimension-repository_nwo" content="d2l-ai/d2l-en" /><meta name="octolytics-dimension-repository_public" content="true" /><meta name="octolytics-dimension-repository_is_fork" content="false" /><meta name="octolytics-dimension-repository_network_root_id" content="152166877" /><meta name="octolytics-dimension-repository_network_root_nwo" content="d2l-ai/d2l-en" /><meta name="octolytics-dimension-repository_explore_github_marketplace_ci_cta_shown" content="false" />


    <link rel="canonical" href="https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py" data-pjax-transient>


  <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">

  <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">

  <link rel="mask-icon" href="https://github.githubassets.com/pinned-octocat.svg" color="#000000">
  <link rel="alternate icon" class="js-site-favicon" type="image/png" href="https://github.githubassets.com/favicons/favicon.png">
  <link rel="icon" class="js-site-favicon" type="image/svg+xml" href="https://github.githubassets.com/favicons/favicon.svg">

<meta name="theme-color" content="#1e2327">


  <link rel="manifest" href="/manifest.json" crossOrigin="use-credentials">

  </head>

  <body class="logged-out env-production page-responsive page-blob">
    

    <div class="position-relative js-header-wrapper ">
      <a href="#start-of-content" class="px-2 py-4 bg-blue text-white show-on-focus js-skip-to-content">Skip to content</a>
      <span class="Progress progress-pjax-loader position-fixed width-full js-pjax-loader-bar">
        <span class="progress-pjax-loader-bar top-0 left-0" style="width: 0%;"></span>
      </span>

      
      



          <header class="Header-old header-logged-out js-details-container Details position-relative f4 py-2" role="banner">
  <div class="container-xl d-lg-flex flex-items-center p-responsive">
    <div class="d-flex flex-justify-between flex-items-center">
        <a class="mr-4" href="https://github.com/" aria-label="Homepage" data-ga-click="(Logged out) Header, go to homepage, icon:logo-wordmark">
          <svg height="32" class="octicon octicon-mark-github text-white" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
        </a>

          <div class="d-lg-none css-truncate css-truncate-target width-fit p-2">
            

          </div>

        <div class="d-flex flex-items-center">
              <a href="/join?ref_cta=Sign+up&amp;ref_loc=header+logged+out&amp;ref_page=%2F%3Cuser-name%3E%2F%3Crepo-name%3E%2Fblob%2Fshow&amp;source=header-repo"
                class="d-inline-block d-lg-none f5 text-white no-underline border border-gray-dark rounded-2 px-2 py-1 mr-3 mr-sm-5"
                data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;site header&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="134b99ae2d02fc84c95ce9ef04304b3a59c755f6cbd3a1e39002304101b592ec"
                data-ga-click="Sign up, click to sign up for account, ref_page:/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show;ref_cta:Sign up;ref_loc:header logged out">
                Sign&nbsp;up
              </a>

          <button class="btn-link d-lg-none mt-1 js-details-target" type="button" aria-label="Toggle navigation" aria-expanded="false">
            <svg height="24" class="octicon octicon-three-bars text-white" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true"><path fill-rule="evenodd" d="M1 2.75A.75.75 0 011.75 2h12.5a.75.75 0 110 1.5H1.75A.75.75 0 011 2.75zm0 5A.75.75 0 011.75 7h12.5a.75.75 0 110 1.5H1.75A.75.75 0 011 7.75zM1.75 12a.75.75 0 100 1.5h12.5a.75.75 0 100-1.5H1.75z"></path></svg>
          </button>
        </div>
    </div>

    <div class="HeaderMenu HeaderMenu--logged-out position-fixed top-0 right-0 bottom-0 height-fit position-lg-relative d-lg-flex flex-justify-between flex-items-center flex-auto">
      <div class="d-flex d-lg-none flex-justify-end border-bottom bg-gray-light p-3">
        <button class="btn-link js-details-target" type="button" aria-label="Toggle navigation" aria-expanded="false">
          <svg height="24" class="octicon octicon-x text-gray" viewBox="0 0 24 24" version="1.1" width="24" aria-hidden="true"><path fill-rule="evenodd" d="M5.72 5.72a.75.75 0 011.06 0L12 10.94l5.22-5.22a.75.75 0 111.06 1.06L13.06 12l5.22 5.22a.75.75 0 11-1.06 1.06L12 13.06l-5.22 5.22a.75.75 0 01-1.06-1.06L10.94 12 5.72 6.78a.75.75 0 010-1.06z"></path></svg>
        </button>
      </div>

        <nav class="mt-0 px-3 px-lg-0 mb-5 mb-lg-0" aria-label="Global">
          <ul class="d-lg-flex list-style-none">
              <li class="d-block d-lg-flex flex-lg-nowrap flex-lg-items-center border-bottom border-lg-bottom-0 mr-0 mr-lg-3 edge-item-fix position-relative flex-wrap flex-justify-between d-flex flex-items-center ">
                <details class="HeaderMenu-details details-overlay details-reset width-full">
                  <summary class="HeaderMenu-summary HeaderMenu-link px-0 py-3 border-0 no-wrap d-block d-lg-inline-block">
                    Why GitHub?
                    <svg x="0px" y="0px" viewBox="0 0 14 8" xml:space="preserve" fill="none" class="icon-chevon-down-mktg position-absolute position-lg-relative">
                      <path d="M1,1l6.2,6L13,1"></path>
                    </svg>
                  </summary>
                  <div class="dropdown-menu flex-auto rounded-1 bg-white px-0 mt-0 pb-4 p-lg-4 position-relative position-lg-absolute left-0 left-lg-n4">
                    <a href="/features" class="py-2 lh-condensed-ultra d-block link-gray-dark no-underline h5 Bump-link--hover" data-ga-click="(Logged out) Header, go to Features">Features <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a>
                    <ul class="list-style-none f5 pb-3">
                      <li class="edge-item-fix"><a href="/features/code-review/" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Code review">Code review</a></li>
                      <li class="edge-item-fix"><a href="/features/project-management/" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Project management">Project management</a></li>
                      <li class="edge-item-fix"><a href="/features/integrations" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Integrations">Integrations</a></li>
                      <li class="edge-item-fix"><a href="/features/actions" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Actions">Actions</a></li>
                      <li class="edge-item-fix"><a href="/features/packages" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to GitHub Packages">Packages</a></li>
                      <li class="edge-item-fix"><a href="/features/security" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Security">Security</a></li>
                      <li class="edge-item-fix"><a href="/features#team-management" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Team management">Team management</a></li>
                      <li class="edge-item-fix"><a href="/features#hosting" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Code hosting">Hosting</a></li>
                      <li class="edge-item-fix hide-xl"><a href="/mobile" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Mobile">Mobile</a></li>
                    </ul>

                    <ul class="list-style-none mb-0 border-lg-top pt-lg-3">
                      <li class="edge-item-fix"><a href="/customer-stories" class="py-2 lh-condensed-ultra d-block no-underline link-gray-dark no-underline h5 Bump-link--hover" data-ga-click="(Logged out) Header, go to Customer stories">Customer stories <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a></li>
                      <li class="edge-item-fix"><a href="/security" class="py-2 lh-condensed-ultra d-block no-underline link-gray-dark no-underline h5 Bump-link--hover" data-ga-click="(Logged out) Header, go to Security">Security <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a></li>
                    </ul>
                  </div>
                </details>
              </li>
              <li class="border-bottom border-lg-bottom-0 mr-0 mr-lg-3">
                <a href="/team" class="HeaderMenu-link no-underline py-3 d-block d-lg-inline-block" data-ga-click="(Logged out) Header, go to Team">Team</a>
              </li>
              <li class="border-bottom border-lg-bottom-0 mr-0 mr-lg-3">
                <a href="/enterprise" class="HeaderMenu-link no-underline py-3 d-block d-lg-inline-block" data-ga-click="(Logged out) Header, go to Enterprise">Enterprise</a>
              </li>

              <li class="d-block d-lg-flex flex-lg-nowrap flex-lg-items-center border-bottom border-lg-bottom-0 mr-0 mr-lg-3 edge-item-fix position-relative flex-wrap flex-justify-between d-flex flex-items-center ">
                <details class="HeaderMenu-details details-overlay details-reset width-full">
                  <summary class="HeaderMenu-summary HeaderMenu-link px-0 py-3 border-0 no-wrap d-block d-lg-inline-block">
                    Explore
                    <svg x="0px" y="0px" viewBox="0 0 14 8" xml:space="preserve" fill="none" class="icon-chevon-down-mktg position-absolute position-lg-relative">
                      <path d="M1,1l6.2,6L13,1"></path>
                    </svg>
                  </summary>

                  <div class="dropdown-menu flex-auto rounded-1 bg-white px-0 pt-2 pb-0 mt-0 pb-4 p-lg-4 position-relative position-lg-absolute left-0 left-lg-n4">
                    <ul class="list-style-none mb-3">
                      <li class="edge-item-fix"><a href="/explore" class="py-2 lh-condensed-ultra d-block link-gray-dark no-underline h5 Bump-link--hover" data-ga-click="(Logged out) Header, go to Explore">Explore GitHub <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a></li>
                    </ul>

                    <h4 class="text-gray-light text-normal text-mono f5 mb-2 border-lg-top pt-lg-3">Learn &amp; contribute</h4>
                    <ul class="list-style-none mb-3">
                      <li class="edge-item-fix"><a href="/topics" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Topics">Topics</a></li>
                        <li class="edge-item-fix"><a href="/collections" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Collections">Collections</a></li>
                      <li class="edge-item-fix"><a href="/trending" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Trending">Trending</a></li>
                      <li class="edge-item-fix"><a href="https://lab.github.com/" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Learning lab">Learning Lab</a></li>
                      <li class="edge-item-fix"><a href="https://opensource.guide" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Open source guides">Open source guides</a></li>
                    </ul>

                    <h4 class="text-gray-light text-normal text-mono f5 mb-2 border-lg-top pt-lg-3">Connect with others</h4>
                    <ul class="list-style-none mb-0">
                      <li class="edge-item-fix"><a href="https://github.com/events" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Events">Events</a></li>
                      <li class="edge-item-fix"><a href="https://github.community" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Community forum">Community forum</a></li>
                      <li class="edge-item-fix"><a href="https://education.github.com" class="py-2 pb-0 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to GitHub Education">GitHub Education</a></li>
                    </ul>
                  </div>
                </details>
              </li>

              <li class="border-bottom border-lg-bottom-0 mr-0 mr-lg-3">
                <a href="/marketplace" class="HeaderMenu-link no-underline py-3 d-block d-lg-inline-block" data-ga-click="(Logged out) Header, go to Marketplace">Marketplace</a>
              </li>

              <li class="d-block d-lg-flex flex-lg-nowrap flex-lg-items-center border-bottom border-lg-bottom-0 mr-0 mr-lg-3 edge-item-fix position-relative flex-wrap flex-justify-between d-flex flex-items-center ">
                <details class="HeaderMenu-details details-overlay details-reset width-full">
                  <summary class="HeaderMenu-summary HeaderMenu-link px-0 py-3 border-0 no-wrap d-block d-lg-inline-block">
                    Pricing
                    <svg x="0px" y="0px" viewBox="0 0 14 8" xml:space="preserve" fill="none" class="icon-chevon-down-mktg position-absolute position-lg-relative">
                       <path d="M1,1l6.2,6L13,1"></path>
                    </svg>
                  </summary>

                  <div class="dropdown-menu flex-auto rounded-1 bg-white px-0 pt-2 pb-4 mt-0 p-lg-4 position-relative position-lg-absolute left-0 left-lg-n4">
                    <a href="/pricing" class="pb-2 lh-condensed-ultra d-block link-gray-dark no-underline h5 Bump-link--hover" data-ga-click="(Logged out) Header, go to Pricing">Plans <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a>

                    <ul class="list-style-none mb-3">
                      <li class="edge-item-fix"><a href="/pricing#feature-comparison" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Compare plans">Compare plans</a></li>
                      <li class="edge-item-fix"><a href="https://enterprise.github.com/contact" class="py-2 lh-condensed-ultra d-block link-gray no-underline f5" data-ga-click="(Logged out) Header, go to Contact Sales">Contact Sales</a></li>
                    </ul>

                    <ul class="list-style-none mb-0 border-lg-top pt-lg-3">
                      <li class="edge-item-fix"><a href="/nonprofit" class="py-2 lh-condensed-ultra d-block no-underline link-gray-dark no-underline h5 Bump-link--hover" data-ga-click="(Logged out) Header, go to Nonprofits">Nonprofit <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a></li>
                      <li class="edge-item-fix"><a href="https://education.github.com" class="py-2 pb-0 lh-condensed-ultra d-block no-underline link-gray-dark no-underline h5 Bump-link--hover"  data-ga-click="(Logged out) Header, go to Education">Education <span class="Bump-link-symbol float-right text-normal text-gray-light">&rarr;</span></a></li>
                    </ul>
                  </div>
                </details>
              </li>
          </ul>
        </nav>

      <div class="d-lg-flex flex-items-center px-3 px-lg-0 text-center text-lg-left">
          <div class="d-lg-flex mb-3 mb-lg-0">
            <div class="header-search header-search-current js-header-search-current flex-auto  flex-self-stretch flex-md-self-auto mr-0 mr-md-3 mb-3 mb-md-0 scoped-search site-scoped-search js-site-search position-relative js-jump-to js-header-search-current-jump-to"
  role="combobox"
  aria-owns="jump-to-results"
  aria-label="Search or jump to"
  aria-haspopup="listbox"
  aria-expanded="false"
>
  <div class="position-relative">
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-site-search-form" role="search" aria-label="Site" data-scope-type="Repository" data-scope-id="152166877" data-scoped-search-url="/d2l-ai/d2l-en/search" data-unscoped-search-url="/search" action="/d2l-ai/d2l-en/search" accept-charset="UTF-8" method="get">
      <label class="form-control input-sm header-search-wrapper p-0 header-search-wrapper-jump-to position-relative d-flex flex-justify-between flex-items-center js-chromeless-input-container">
        <input type="text"
          class="form-control input-sm header-search-input jump-to-field js-jump-to-field js-site-search-focus js-site-search-field is-clearable"
          data-hotkey="s,/"
          name="q"
          value=""
          placeholder="Search"
          data-unscoped-placeholder="Search GitHub"
          data-scoped-placeholder="Search"
          autocapitalize="off"
          aria-autocomplete="list"
          aria-controls="jump-to-results"
          aria-label="Search"
          data-jump-to-suggestions-path="/_graphql/GetSuggestedNavigationDestinations"
          spellcheck="false"
          autocomplete="off"
          >
          <input type="hidden" data-csrf="true" class="js-data-jump-to-suggestions-path-csrf" value="VoalEagPsKIg3MeMwEFAqCbpup6MA5KR2YGx9751f69W6PgxHclqBpKzbjQ1R/1CkyrmBl9+Vcv73+G/QepReA==" />
          <input type="hidden" class="js-site-search-type-field" name="type" >
            <img src="https://github.githubassets.com/images/search-key-slash.svg" alt="" class="mr-2 header-search-key-slash">

            <div class="Box position-absolute overflow-hidden d-none jump-to-suggestions js-jump-to-suggestions-container">
              
<ul class="d-none js-jump-to-suggestions-template-container">
  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-suggestion" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M1.75 0A1.75 1.75 0 000 1.75v12.5C0 15.216.784 16 1.75 16h12.5A1.75 1.75 0 0016 14.25V1.75A1.75 1.75 0 0014.25 0H1.75zM1.5 1.75a.25.25 0 01.25-.25h12.5a.25.25 0 01.25.25v12.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25V1.75zM11.75 3a.75.75 0 00-.75.75v7.5a.75.75 0 001.5 0v-7.5a.75.75 0 00-.75-.75zm-8.25.75a.75.75 0 011.5 0v5.5a.75.75 0 01-1.5 0v-5.5zM8 3a.75.75 0 00-.75.75v3.5a.75.75 0 001.5 0v-3.5A.75.75 0 008 3z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M11.5 7a4.499 4.499 0 11-8.998 0A4.499 4.499 0 0111.5 7zm-.82 4.74a6 6 0 111.06-1.06l3.04 3.04a.75.75 0 11-1.06 1.06l-3.04-3.04z"></path></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>

</ul>

<ul class="d-none js-jump-to-no-results-template-container">
  <li class="d-flex flex-justify-center flex-items-center f5 d-none js-jump-to-suggestion p-2">
    <span class="text-gray">No suggested jump to results</span>
  </li>
</ul>

<ul id="jump-to-results" role="listbox" class="p-0 m-0 js-navigation-container jump-to-suggestions-results-container js-jump-to-suggestions-results-container">
  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-scoped-search d-none" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M1.75 0A1.75 1.75 0 000 1.75v12.5C0 15.216.784 16 1.75 16h12.5A1.75 1.75 0 0016 14.25V1.75A1.75 1.75 0 0014.25 0H1.75zM1.5 1.75a.25.25 0 01.25-.25h12.5a.25.25 0 01.25.25v12.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25V1.75zM11.75 3a.75.75 0 00-.75.75v7.5a.75.75 0 001.5 0v-7.5a.75.75 0 00-.75-.75zm-8.25.75a.75.75 0 011.5 0v5.5a.75.75 0 01-1.5 0v-5.5zM8 3a.75.75 0 00-.75.75v3.5a.75.75 0 001.5 0v-3.5A.75.75 0 008 3z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M11.5 7a4.499 4.499 0 11-8.998 0A4.499 4.499 0 0111.5 7zm-.82 4.74a6 6 0 111.06-1.06l3.04 3.04a.75.75 0 11-1.06 1.06l-3.04-3.04z"></path></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>

  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-global-search d-none" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M1.75 0A1.75 1.75 0 000 1.75v12.5C0 15.216.784 16 1.75 16h12.5A1.75 1.75 0 0016 14.25V1.75A1.75 1.75 0 0014.25 0H1.75zM1.5 1.75a.25.25 0 01.25-.25h12.5a.25.25 0 01.25.25v12.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25V1.75zM11.75 3a.75.75 0 00-.75.75v7.5a.75.75 0 001.5 0v-7.5a.75.75 0 00-.75-.75zm-8.25.75a.75.75 0 011.5 0v5.5a.75.75 0 01-1.5 0v-5.5zM8 3a.75.75 0 00-.75.75v3.5a.75.75 0 001.5 0v-3.5A.75.75 0 008 3z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M11.5 7a4.499 4.499 0 11-8.998 0A4.499 4.499 0 0111.5 7zm-.82 4.74a6 6 0 111.06-1.06l3.04 3.04a.75.75 0 11-1.06 1.06l-3.04-3.04z"></path></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>


</ul>

            </div>
      </label>
</form>  </div>
</div>

          </div>

        <a href="/login?return_to=%2Fd2l-ai%2Fd2l-en%2Fblob%2Fmaster%2Fd2l%2Ftorch.py"
          class="HeaderMenu-link no-underline mr-3"
          data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;site header menu&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="1c2a1365470550d75cd61b44a0dd70bd2e2033c22a3093da5d72e062fc0d7ab8"
          data-ga-click="(Logged out) Header, clicked Sign in, text:sign-in">
          Sign&nbsp;in
        </a>
            <a href="/join?ref_cta=Sign+up&amp;ref_loc=header+logged+out&amp;ref_page=%2F%3Cuser-name%3E%2F%3Crepo-name%3E%2Fblob%2Fshow&amp;source=header-repo&amp;source_repo=d2l-ai%2Fd2l-en"
              class="HeaderMenu-link d-inline-block no-underline border border-gray-dark rounded-1 px-2 py-1"
              data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;site header menu&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="1c2a1365470550d75cd61b44a0dd70bd2e2033c22a3093da5d72e062fc0d7ab8"
              data-ga-click="Sign up, click to sign up for account, ref_page:/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show;ref_cta:Sign up;ref_loc:header logged out">
              Sign&nbsp;up
            </a>
      </div>
    </div>
  </div>
</header>

    </div>

  <div id="start-of-content" class="show-on-focus"></div>




    <div id="js-flash-container">


  <template class="js-flash-template">
    <div class="flash flash-full  js-flash-template-container">
  <div class=" px-2" >
    <button class="flash-close js-flash-close" type="button" aria-label="Dismiss this message">
      <svg class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
    </button>
    
      <div class="js-flash-template-message"></div>

  </div>
</div>
  </template>
</div>


  

  <include-fragment class="js-notification-shelf-include-fragment" data-base-src="https://github.com/notifications/beta/shelf"></include-fragment>



  <div
    class="application-main "
    data-commit-hovercards-enabled
    data-discussion-hovercards-enabled
    data-issue-and-pr-hovercards-enabled
  >
        <div itemscope itemtype="http://schema.org/SoftwareSourceCode" class="">
    <main  >
      

  


  










  <div class="bg-gray-light pt-3 hide-full-screen mb-5">

    <div class="d-flex mb-3 px-3 px-md-4 px-lg-5">

      <div class="flex-auto min-width-0 width-fit mr-3">
        <h1 class=" d-flex flex-wrap flex-items-center break-word f3 text-normal">
    <svg class="octicon octicon-repo text-gray" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"></path></svg>
  <span class="author ml-2 flex-self-stretch" itemprop="author">
    <a class="url fn" rel="author" data-hovercard-type="organization" data-hovercard-url="/orgs/d2l-ai/hovercard" href="/d2l-ai">d2l-ai</a>
  </span>
  <span class="mx-1 flex-self-stretch">/</span>
  <strong itemprop="name" class="mr-2 flex-self-stretch">
    <a data-pjax="#js-repo-pjax-container" href="/d2l-ai/d2l-en">d2l-en</a>
  </strong>
  
</h1>


      </div>

      <ul class="pagehead-actions flex-shrink-0 d-none d-md-inline" style="padding: 2px 0;">

  <li>
      <a class="tooltipped tooltipped-s btn btn-sm btn-with-count" aria-label="You must be signed in to watch a repository" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;notification subscription menu watch&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;LOG_IN&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="4b009c7240548a6664717072b3cd4fe2f36ba1e4c8712c2c42bc0f2c292a75d7" href="/login?return_to=%2Fd2l-ai%2Fd2l-en">
    <svg height="16" class="octicon octicon-eye" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.679 7.932c.412-.621 1.242-1.75 2.366-2.717C5.175 4.242 6.527 3.5 8 3.5c1.473 0 2.824.742 3.955 1.715 1.124.967 1.954 2.096 2.366 2.717a.119.119 0 010 .136c-.412.621-1.242 1.75-2.366 2.717C10.825 11.758 9.473 12.5 8 12.5c-1.473 0-2.824-.742-3.955-1.715C2.92 9.818 2.09 8.69 1.679 8.068a.119.119 0 010-.136zM8 2c-1.981 0-3.67.992-4.933 2.078C1.797 5.169.88 6.423.43 7.1a1.619 1.619 0 000 1.798c.45.678 1.367 1.932 2.637 3.024C4.329 13.008 6.019 14 8 14c1.981 0 3.67-.992 4.933-2.078 1.27-1.091 2.187-2.345 2.637-3.023a1.619 1.619 0 000-1.798c-.45-.678-1.367-1.932-2.637-3.023C11.671 2.992 9.981 2 8 2zm0 8a2 2 0 100-4 2 2 0 000 4z"></path></svg>
    Watch
</a>    <a class="social-count" href="/d2l-ai/d2l-en/watchers"
       aria-label="225 users are watching this repository">
      225
    </a>

  </li>

  <li>
        <a class="btn btn-sm btn-with-count  tooltipped tooltipped-s" aria-label="You must be signed in to star a repository" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;star button&quot;,&quot;repository_id&quot;:152166877,&quot;auth_type&quot;:&quot;LOG_IN&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="deb20bc5793187e2784d6a95d29e853a125b893197fb9e9ebc7db492ec305039" href="/login?return_to=%2Fd2l-ai%2Fd2l-en">
      <svg vertical_align="text_bottom" height="16" class="octicon octicon-star v-align-text-bottom" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25zm0 2.445L6.615 5.5a.75.75 0 01-.564.41l-3.097.45 2.24 2.184a.75.75 0 01.216.664l-.528 3.084 2.769-1.456a.75.75 0 01.698 0l2.77 1.456-.53-3.084a.75.75 0 01.216-.664l2.24-2.183-3.096-.45a.75.75 0 01-.564-.41L8 2.694v.001z"></path></svg>
      Star
</a>
    <a class="social-count js-social-count" href="/d2l-ai/d2l-en/stargazers"
      aria-label="6662 users starred this repository">
      6.7k
    </a>

  </li>

  <li>
      <a class="btn btn-sm btn-with-count tooltipped tooltipped-s" aria-label="You must be signed in to fork a repository" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;repo details fork button&quot;,&quot;repository_id&quot;:152166877,&quot;auth_type&quot;:&quot;LOG_IN&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="9bf8018729dd608715e707eca218628c3c87363f607fc8109fb4aa70b49be3d9" href="/login?return_to=%2Fd2l-ai%2Fd2l-en">
        <svg class="octicon octicon-repo-forked" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M5 3.25a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm0 2.122a2.25 2.25 0 10-1.5 0v.878A2.25 2.25 0 005.75 8.5h1.5v2.128a2.251 2.251 0 101.5 0V8.5h1.5a2.25 2.25 0 002.25-2.25v-.878a2.25 2.25 0 10-1.5 0v.878a.75.75 0 01-.75.75h-4.5A.75.75 0 015 6.25v-.878zm3.75 7.378a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm3-8.75a.75.75 0 100-1.5.75.75 0 000 1.5z"></path></svg>
        Fork
</a>
    <a href="/d2l-ai/d2l-en/network/members" class="social-count"
       aria-label="1511 users forked this repository">
      1.5k
    </a>
  </li>
</ul>

    </div>
    
<nav class="js-repo-nav js-sidenav-container-pjax js-responsive-underlinenav overflow-hidden UnderlineNav px-3 px-md-4 px-lg-5 bg-gray-light" aria-label="Repository" data-pjax="#js-repo-pjax-container">
  <ul class="UnderlineNav-body list-style-none ">
          <li class="d-flex">
        <a class="js-selected-navigation-item selected UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="code-tab" data-hotkey="g c" data-ga-click="Repository, Navigation click, Code tab" aria-current="page" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages repo_deployments /d2l-ai/d2l-en" href="/d2l-ai/d2l-en">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-code UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M4.72 3.22a.75.75 0 011.06 1.06L2.06 8l3.72 3.72a.75.75 0 11-1.06 1.06L.47 8.53a.75.75 0 010-1.06l4.25-4.25zm6.56 0a.75.75 0 10-1.06 1.06L13.94 8l-3.72 3.72a.75.75 0 101.06 1.06l4.25-4.25a.75.75 0 000-1.06l-4.25-4.25z"></path></svg>
            <span data-content="Code">Code</span>
              <span class="Counter " title="Not available"></span>
</a>      </li>
      <li class="d-flex">
        <a class="js-selected-navigation-item UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="issues-tab" data-hotkey="g i" data-ga-click="Repository, Navigation click, Issues tab" data-selected-links="repo_issues repo_labels repo_milestones /d2l-ai/d2l-en/issues" href="/d2l-ai/d2l-en/issues">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-issue-opened UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 1.5a6.5 6.5 0 100 13 6.5 6.5 0 000-13zM0 8a8 8 0 1116 0A8 8 0 010 8zm9 3a1 1 0 11-2 0 1 1 0 012 0zm-.25-6.25a.75.75 0 00-1.5 0v3.5a.75.75 0 001.5 0v-3.5z"></path></svg>
            <span data-content="Issues">Issues</span>
              <span class="Counter " title="24">24</span>
</a>      </li>
      <li class="d-flex">
        <a class="js-selected-navigation-item UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="pull-requests-tab" data-hotkey="g p" data-ga-click="Repository, Navigation click, Pull requests tab" data-selected-links="repo_pulls checks /d2l-ai/d2l-en/pulls" href="/d2l-ai/d2l-en/pulls">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-git-pull-request UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.177 3.073L9.573.677A.25.25 0 0110 .854v4.792a.25.25 0 01-.427.177L7.177 3.427a.25.25 0 010-.354zM3.75 2.5a.75.75 0 100 1.5.75.75 0 000-1.5zm-2.25.75a2.25 2.25 0 113 2.122v5.256a2.251 2.251 0 11-1.5 0V5.372A2.25 2.25 0 011.5 3.25zM11 2.5h-1V4h1a1 1 0 011 1v5.628a2.251 2.251 0 101.5 0V5A2.5 2.5 0 0011 2.5zm1 10.25a.75.75 0 111.5 0 .75.75 0 01-1.5 0zM3.75 12a.75.75 0 100 1.5.75.75 0 000-1.5z"></path></svg>
            <span data-content="Pull requests">Pull requests</span>
              <span class="Counter " title="10">10</span>
</a>      </li>
      <li class="d-flex">
        <a class="js-selected-navigation-item UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="actions-tab" data-hotkey="g w" data-ga-click="Repository, Navigation click, Actions tab" data-selected-links="repo_actions /d2l-ai/d2l-en/actions" href="/d2l-ai/d2l-en/actions">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-play UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0zM8 0a8 8 0 100 16A8 8 0 008 0zM6.379 5.227A.25.25 0 006 5.442v5.117a.25.25 0 00.379.214l4.264-2.559a.25.25 0 000-.428L6.379 5.227z"></path></svg>
            <span data-content="Actions">Actions</span>
              <span class="Counter " title="Not available"></span>
</a>      </li>
      <li class="d-flex">
        <a class="js-selected-navigation-item UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="security-tab" data-hotkey="g s" data-ga-click="Repository, Navigation click, Security tab" data-selected-links="security overview alerts policy token_scanning code_scanning /d2l-ai/d2l-en/security" href="/d2l-ai/d2l-en/security">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-shield UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.467.133a1.75 1.75 0 011.066 0l5.25 1.68A1.75 1.75 0 0115 3.48V7c0 1.566-.32 3.182-1.303 4.682-.983 1.498-2.585 2.813-5.032 3.855a1.7 1.7 0 01-1.33 0c-2.447-1.042-4.049-2.357-5.032-3.855C1.32 10.182 1 8.566 1 7V3.48a1.75 1.75 0 011.217-1.667l5.25-1.68zm.61 1.429a.25.25 0 00-.153 0l-5.25 1.68a.25.25 0 00-.174.238V7c0 1.358.275 2.666 1.057 3.86.784 1.194 2.121 2.34 4.366 3.297a.2.2 0 00.154 0c2.245-.956 3.582-2.104 4.366-3.298C13.225 9.666 13.5 8.36 13.5 7V3.48a.25.25 0 00-.174-.237l-5.25-1.68zM9 10.5a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.75a.75.75 0 10-1.5 0v3a.75.75 0 001.5 0v-3z"></path></svg>
            <span data-content="Security">Security</span>
              <span class="js-security-tab-count Counter " data-url="/d2l-ai/d2l-en/security/overall-count" title="Not available"></span>
</a>      </li>
      <li class="d-flex">
        <a class="js-selected-navigation-item UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="insights-tab" data-ga-click="Repository, Navigation click, Insights tab" data-selected-links="repo_graphs repo_contributors dependency_graph dependabot_updates pulse people /d2l-ai/d2l-en/pulse" href="/d2l-ai/d2l-en/pulse">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-graph UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 1.75a.75.75 0 00-1.5 0v12.5c0 .414.336.75.75.75h14.5a.75.75 0 000-1.5H1.5V1.75zm14.28 2.53a.75.75 0 00-1.06-1.06L10 7.94 7.53 5.47a.75.75 0 00-1.06 0L3.22 8.72a.75.75 0 001.06 1.06L7 7.06l2.47 2.47a.75.75 0 001.06 0l5.25-5.25z"></path></svg>
            <span data-content="Insights">Insights</span>
              <span class="Counter " title="Not available"></span>
</a>      </li>

</ul>        <div class="position-absolute right-0 pr-3 pr-md-4 pr-lg-5 js-responsive-underlinenav-overflow" style="visibility:hidden;">
      <details class="details-overlay details-reset position-relative">
  <summary role="button">
              <div class="UnderlineNav-item mr-0 border-0">
            <svg class="octicon octicon-kebab-horizontal" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="M8 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM1.5 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm13 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3z"></path></svg>
            <span class="sr-only">More</span>
          </div>

</summary>            <details-menu class="dropdown-menu dropdown-menu-sw " role="menu">
  
            <ul>
                <li data-menu-item="code-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /d2l-ai/d2l-en" href="/d2l-ai/d2l-en">
                    Code
</a>                </li>
                <li data-menu-item="issues-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /d2l-ai/d2l-en/issues" href="/d2l-ai/d2l-en/issues">
                    Issues
</a>                </li>
                <li data-menu-item="pull-requests-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /d2l-ai/d2l-en/pulls" href="/d2l-ai/d2l-en/pulls">
                    Pull requests
</a>                </li>
                <li data-menu-item="actions-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /d2l-ai/d2l-en/actions" href="/d2l-ai/d2l-en/actions">
                    Actions
</a>                </li>
                <li data-menu-item="security-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /d2l-ai/d2l-en/security" href="/d2l-ai/d2l-en/security">
                    Security
</a>                </li>
                <li data-menu-item="insights-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /d2l-ai/d2l-en/pulse" href="/d2l-ai/d2l-en/pulse">
                    Insights
</a>                </li>
            </ul>

</details-menu>
</details>    </div>

</nav>
  </div>

<div class="container-xl clearfix new-discussion-timeline  px-3 px-md-4 px-lg-5">
  <div class="repository-content " >

    
    

  


    <a class="d-none js-permalink-shortcut" data-hotkey="y" href="/d2l-ai/d2l-en/blob/9607f62e2bb336e25f102347d44907e3f69e567a/d2l/torch.py">Permalink</a>

    <!-- blob contrib key: blob_contributors:v22:f1ef1abcaeb25748483f1b7353e33ed8 -->
      <signup-prompt class="signup-prompt-bg rounded-1" data-prompt="signup" hidden>
    <div class="signup-prompt p-4 text-center mb-4 rounded-1">
      <div class="position-relative">
        <button
          type="button"
          class="position-absolute top-0 right-0 btn-link link-gray"
          data-action="click:signup-prompt#dismiss"
          data-ga-click="(Logged out) Sign up prompt, clicked Dismiss, text:dismiss"
        >
          Dismiss
        </button>
        <h3 class="pt-2">Join GitHub today</h3>
        <p class="col-6 mx-auto">GitHub is home to over 50 million developers working together to host and review code, manage projects, and build software together.</p>
        <a class="btn btn-primary" data-ga-click="(Logged out) Sign up prompt, clicked Sign up, text:sign-up" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;files signup prompt&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="5a3242d072ebf9b2e0312b70a4c5a9a58756da2ce6e29c7e6b5c381cd3771d2b" href="/join?source=prompt-blob-show&amp;source_repo=d2l-ai%2Fd2l-en">Sign up</a>
      </div>
    </div>
  </signup-prompt>


    <div class="d-flex flex-items-start flex-shrink-0 pb-3 flex-wrap flex-md-nowrap flex-justify-between flex-md-justify-start">
      
<details class="details-reset details-overlay mr-0 mb-0 " id="branch-select-menu">
  <summary class="btn css-truncate"
           data-hotkey="w"
           title="Switch branches or tags">
    <svg text="gray" height="16" class="octicon octicon-git-branch text-gray" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M11.75 2.5a.75.75 0 100 1.5.75.75 0 000-1.5zm-2.25.75a2.25 2.25 0 113 2.122V6A2.5 2.5 0 0110 8.5H6a1 1 0 00-1 1v1.128a2.251 2.251 0 11-1.5 0V5.372a2.25 2.25 0 111.5 0v1.836A2.492 2.492 0 016 7h4a1 1 0 001-1v-.628A2.25 2.25 0 019.5 3.25zM4.25 12a.75.75 0 100 1.5.75.75 0 000-1.5zM3.5 3.25a.75.75 0 111.5 0 .75.75 0 01-1.5 0z"></path></svg>
    <span class="css-truncate-target" data-menu-button>master</span>
    <span class="dropdown-caret"></span>
  </summary>

  <details-menu class="SelectMenu SelectMenu--hasFilter" src="/d2l-ai/d2l-en/refs/master/d2l/torch.py?source_action=show&amp;source_controller=blob" preload>
    <div class="SelectMenu-modal">
      <include-fragment class="SelectMenu-loading" aria-label="Menu is loading">
        <svg class="octicon octicon-octoface anim-pulse" height="32" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M14.7 5.34c.13-.32.55-1.59-.13-3.31 0 0-1.05-.33-3.44 1.3-1-.28-2.07-.32-3.13-.32s-2.13.04-3.13.32c-2.39-1.64-3.44-1.3-3.44-1.3-.68 1.72-.26 2.99-.13 3.31C.49 6.21 0 7.33 0 8.69 0 13.84 3.33 15 7.98 15S16 13.84 16 8.69c0-1.36-.49-2.48-1.3-3.35zM8 14.02c-3.3 0-5.98-.15-5.98-3.35 0-.76.38-1.48 1.02-2.07 1.07-.98 2.9-.46 4.96-.46 2.07 0 3.88-.52 4.96.46.65.59 1.02 1.3 1.02 2.07 0 3.19-2.68 3.35-5.98 3.35zM5.49 9.01c-.66 0-1.2.8-1.2 1.78s.54 1.79 1.2 1.79c.66 0 1.2-.8 1.2-1.79s-.54-1.78-1.2-1.78zm5.02 0c-.66 0-1.2.79-1.2 1.78s.54 1.79 1.2 1.79c.66 0 1.2-.8 1.2-1.79s-.53-1.78-1.2-1.78z"></path></svg>
      </include-fragment>
    </div>
  </details-menu>
</details>

      <h2 id="blob-path" class="breadcrumb flex-auto min-width-0 text-normal mx-0 mx-md-3 width-full width-md-auto flex-order-1 flex-md-order-none mt-3 mt-md-0">
        <span class="js-repo-root text-bold"><span class="js-path-segment d-inline-block wb-break-all"><a data-pjax="true" href="/d2l-ai/d2l-en"><span>d2l-en</span></a></span></span><span class="separator">/</span><span class="js-path-segment d-inline-block wb-break-all"><a data-pjax="true" href="/d2l-ai/d2l-en/tree/master/d2l"><span>d2l</span></a></span><span class="separator">/</span><strong class="final-path">torch.py</strong>
          <span class="separator">/</span><details class="details-reset details-overlay d-inline" id="jumpto-symbol-select-menu">
  <summary class="btn-link link-gray css-truncate" aria-haspopup="true" data-hotkey="r" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.click_on_blob_definitions&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;click_on_blob_definitions&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="3769cc80dc5bf08ecce2cae7b04ffa7541ab8bc6609a6d0728d8f6940ac28426">
      <svg class="octicon octicon-code" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4.72 3.22a.75.75 0 011.06 1.06L2.06 8l3.72 3.72a.75.75 0 11-1.06 1.06L.47 8.53a.75.75 0 010-1.06l4.25-4.25zm6.56 0a.75.75 0 10-1.06 1.06L13.94 8l-3.72 3.72a.75.75 0 101.06 1.06l4.25-4.25a.75.75 0 000-1.06l-4.25-4.25z"></path></svg>
    <span data-menu-button>Jump to</span>
    <span class="dropdown-caret"></span>
  </summary>
  <details-menu class="SelectMenu SelectMenu--hasFilter" role="menu">
    <div class="SelectMenu-modal">
      <header class="SelectMenu-header">
        <span class="SelectMenu-title">Code definitions</span>
        <button class="SelectMenu-closeButton" type="button" data-toggle-for="jumpto-symbol-select-menu">
          <svg aria-label="Close menu" class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" width="16" height="16" role="img"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
        </button>
      </header>
        <div class="SelectMenu-filter">
          <input
            class="SelectMenu-input form-control js-filterable-field"
            id="jumpto-symbols-filter-field"
            type="text"
            autocomplete="off"
            spellcheck="false"
            autofocus
            placeholder="Filter definitions"
            aria-label="Filter definitions">
        </div>
      <div class="SelectMenu-list">
        <div data-filterable-for="jumpto-symbols-filter-field" data-filterable-type="substring">
            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L36">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>mkdir_if_not_exist</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L45">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>use_svg_display</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L51">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>set_figsize</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L58">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>set_axes</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L72">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>plot</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L83">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>has_one_axis</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L105">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>Timer</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L107">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L111">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>start</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L115">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>stop</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L120">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>avg</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L124">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>sum</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L128">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>cumsum</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L134">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>synthetic_data</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L143">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>linreg</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L149">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>squared_loss</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L155">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>sgd</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L163">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>load_array</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L170">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>get_fashion_mnist_labels</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L178">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>show_images</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L193">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>get_dataloader_workers</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L199">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>load_data_fashion_mnist</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L216">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>accuracy</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L225">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>evaluate_accuracy</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L236">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>Accumulator</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L238">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L241">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>add</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L244">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>reset</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L247">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__getitem__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L252">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>train_epoch_ch3</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L278">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>Animator</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L280">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L296">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>add</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L320">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>train_ch3</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L329">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>assert</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L330">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>assert</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L331">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>assert</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L335">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>predict_ch3</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L346">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>evaluate_loss</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L367">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>download</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L369">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>assert</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L391">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>download_extract</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L401">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>assert</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L407">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>download_all</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L426">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>try_gpu</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L434">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>try_all_gpus</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L442">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>corr2d</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L453">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>evaluate_accuracy_gpu</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L467">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>train_ch6</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L470">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>init_weights</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L510">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>Residual</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L511">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L527">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>forward</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L542">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>read_time_machine</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L551">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>tokenize</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L562">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>Vocab</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L563">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L578">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__len__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L581">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__getitem__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L586">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>to_tokens</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L593">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>count_corpus</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L600">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>load_corpus_time_machine</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L611">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>seq_data_iter_random</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L619">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>data</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L634">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>seq_data_iter_consecutive</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L650">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>SeqDataLoader</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L652">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L660">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__iter__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L665">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>load_data_time_machine</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L673">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>RNNModelScratch</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L675">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L681">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__call__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L685">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>begin_state</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L690">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>predict_ch8</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L705">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>grad_clipping</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L717">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>train_epoch_ch8</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L747">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>train_ch8</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L777">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>read_data_nmt</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L784">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>preprocess_nmt</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L785">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>no_space</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L795">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>tokenize_nmt</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L808">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>truncate_pad</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L815">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>build_array</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L826">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>load_data_nmt</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L843">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>Encoder</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L845">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L848">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>forward</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L853">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>Decoder</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L855">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L858">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>init_state</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L861">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>forward</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L866">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>EncoderDecoder</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L868">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L873">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>forward</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L880">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>Seq2SeqEncoder</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L881">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L887">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>forward</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L899">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>Seq2SeqDecoder</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L900">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L907">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>init_state</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L910">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>forward</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L919">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>sequence_mask</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L927">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>MaskedSoftmaxCELoss</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L931">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>forward</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L941">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>train_s2s_ch9</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L942">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>xavier_init_weights</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L977">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>predict_s2s_ch9</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1001">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>masked_softmax</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1019">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>DotProductAttention</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1020">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1028">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>forward</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1037">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>MLPAttention</span>
              <span class="flex-auto d-flex flex-justify-end">Class</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1038">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>__init__</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1045">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>forward</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1057">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>annotate</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1063">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>train_2d</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1076">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>show_trace_2d</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1093">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>get_data_ch11</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1103">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>train_ch11</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1130">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>train_concise_ch11</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>            <a class="SelectMenu-item d-flex flex-justify-between css-truncate" role="menuitemradio" aria-checked="false" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.navigate_to_blob_definition&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;navigate_to_blob_definition&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="bbdb15ce67afe06b8e5554000e7db59927d7a10a1f83ab0e9f25ba22e651a544" href="/d2l-ai/d2l-en/blob/master/d2l/torch.py#L1133">
              <svg class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <span class="flex-auto css-truncate-target" data-menu-button-text>init_weights</span>
              <span class="flex-auto d-flex flex-justify-end">Function</span>
</a>        </div>
      </div>
      <footer class="SelectMenu-footer">
        <div class="d-flex flex-justify-between">
          Code navigation index up-to-date
          <svg class="octicon octicon-dot-fill text-green" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8z"></path></svg>
        </div>
      </footer>
    </div>
  </details-menu>
</details>

      </h2>
      <a href="/d2l-ai/d2l-en/find/master"
            class="js-pjax-capture-input btn mr-2 d-none d-md-block"
            data-pjax
            data-hotkey="t">
        Go to file
      </a>

      <details class="details-overlay details-reset position-relative" id="blob-more-options-details">
  <summary role="button">
              <span class="btn">
            <svg aria-label="More options" height="16" class="octicon octicon-kebab-horizontal" viewBox="0 0 16 16" version="1.1" width="16" role="img"><path d="M8 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM1.5 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm13 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3z"></path></svg>
          </span>

</summary>            <ul class="dropdown-menu dropdown-menu-sw">
            <li class="d-block d-md-none">
              <a class="dropdown-item d-flex flex-items-baseline" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;FIND_FILE_BUTTON&quot;,&quot;repository_id&quot;:152166877,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="1c143a09a52c777c57c170bd24081185143142819f870d86e8902790a7fc1a67" data-ga-click="Repository, find file, location:repo overview" data-hotkey="t" data-pjax="true" href="/d2l-ai/d2l-en/find/master">
                <span class="flex-auto">Go to file</span>
                <span class="text-small text-gray" aria-hidden="true">T</span>
</a>            </li>
            <li data-toggle-for="blob-more-options-details">
              <button type="button" data-toggle-for="jumpto-line-details-dialog" class="btn-link dropdown-item">
                <span class="d-flex flex-items-baseline">
                  <span class="flex-auto">Go to line</span>
                  <span class="text-small text-gray" aria-hidden="true">L</span>
                </span>
              </button>
            </li>
            <li data-toggle-for="blob-more-options-details">
              <button type="button" data-toggle-for="jumpto-symbol-select-menu" class="btn-link dropdown-item">
                <span class="d-flex flex-items-baseline">
                  <span class="flex-auto">Go to definition</span>
                  <span class="text-small text-gray" aria-hidden="true">R</span>
                </span>
              </button>
            </li>
            <li class="dropdown-divider" role="none"></li>
            <li>
              <clipboard-copy value="d2l/torch.py" class="dropdown-item cursor-pointer" data-toggle-for="blob-more-options-details">
                Copy path
              </clipboard-copy>
            </li>
          </ul>

</details>    </div>



    <div class="Box d-flex flex-column flex-shrink-0 mb-3">
      <include-fragment src="/d2l-ai/d2l-en/contributors/master/d2l/torch.py" class="commit-loader">
        <div class="Box-header Box-header--blue d-flex flex-items-center">
          <div class="Skeleton avatar avatar-user flex-shrink-0 ml-n1 mr-n1 mt-n1 mb-n1" style="width:24px;height:24px;"></div>
          <div class="Skeleton Skeleton--text col-5 ml-2">&nbsp;</div>
        </div>

        <div class="Box-body d-flex flex-items-center" >
          <div class="Skeleton Skeleton--text col-1">&nbsp;</div>
          <span class="text-red h6 loader-error">Cannot retrieve contributors at this time</span>
        </div>
</include-fragment>    </div>






    <div class="Box mt-3 position-relative
      ">
      
<div class="Box-header py-2 d-flex flex-column flex-shrink-0 flex-md-row flex-md-items-center">
  <div class="text-mono f6 flex-auto pr-3 flex-order-2 flex-md-order-1 mt-2 mt-md-0">

      1196 lines (1000 sloc)
      <span class="file-info-divider"></span>
    44.9 KB
  </div>

  <div class="d-flex py-1 py-md-0 flex-auto flex-order-1 flex-md-order-2 flex-sm-grow-0 flex-justify-between">

    <div class="BtnGroup">
      <a class="btn btn-sm BtnGroup-item " href="/d2l-ai/d2l-en/raw/master/d2l/torch.py" id="raw-url" role="button">Raw</a>
        <a class="btn js-update-url-with-hash btn-sm BtnGroup-item " href="/d2l-ai/d2l-en/blame/master/d2l/torch.py" data-hotkey="b" role="button">Blame</a>
    </div>

    <div>
          <a class="btn-octicon tooltipped tooltipped-nw js-remove-unless-platform"
             data-platforms="windows,mac"
             href="https://desktop.github.com"
             aria-label="Open this file in GitHub Desktop"
             data-ga-click="Repository, open with desktop">
              <svg class="octicon octicon-device-desktop" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.75 2.5h12.5a.25.25 0 01.25.25v7.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25v-7.5a.25.25 0 01.25-.25zM14.25 1H1.75A1.75 1.75 0 000 2.75v7.5C0 11.216.784 12 1.75 12h3.727c-.1 1.041-.52 1.872-1.292 2.757A.75.75 0 004.75 16h6.5a.75.75 0 00.565-1.243c-.772-.885-1.193-1.716-1.292-2.757h3.727A1.75 1.75 0 0016 10.25v-7.5A1.75 1.75 0 0014.25 1zM9.018 12H6.982a5.72 5.72 0 01-.765 2.5h3.566a5.72 5.72 0 01-.765-2.5z"></path></svg>
          </a>

          <button type="button" class="btn-octicon disabled tooltipped tooltipped-nw"
            aria-label="You must be signed in to make or propose changes">
            <svg class="octicon octicon-pencil" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M11.013 1.427a1.75 1.75 0 012.474 0l1.086 1.086a1.75 1.75 0 010 2.474l-8.61 8.61c-.21.21-.47.364-.756.445l-3.251.93a.75.75 0 01-.927-.928l.929-3.25a1.75 1.75 0 01.445-.758l8.61-8.61zm1.414 1.06a.25.25 0 00-.354 0L10.811 3.75l1.439 1.44 1.263-1.263a.25.25 0 000-.354l-1.086-1.086zM11.189 6.25L9.75 4.81l-6.286 6.287a.25.25 0 00-.064.108l-.558 1.953 1.953-.558a.249.249 0 00.108-.064l6.286-6.286z"></path></svg>
          </button>
          <button type="button" class="btn-octicon btn-octicon-danger disabled tooltipped tooltipped-nw"
            aria-label="You must be signed in to make or propose changes">
            <svg class="octicon octicon-trashcan" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M6.5 1.75a.25.25 0 01.25-.25h2.5a.25.25 0 01.25.25V3h-3V1.75zm4.5 0V3h2.25a.75.75 0 010 1.5H2.75a.75.75 0 010-1.5H5V1.75C5 .784 5.784 0 6.75 0h2.5C10.216 0 11 .784 11 1.75zM4.496 6.675a.75.75 0 10-1.492.15l.66 6.6A1.75 1.75 0 005.405 15h5.19c.9 0 1.652-.681 1.741-1.576l.66-6.6a.75.75 0 00-1.492-.149l-.66 6.6a.25.25 0 01-.249.225h-5.19a.25.25 0 01-.249-.225l-.66-6.6z"></path></svg>
          </button>
    </div>
  </div>
</div>



      

  <div itemprop="text" class="Box-body p-0 blob-wrapper data type-python ">
      
<table class="highlight tab-size js-file-line-container" data-tab-size="8" data-paste-markdown-skip>
      <tr>
        <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
        <td id="LC1" class="blob-code blob-code-inner js-file-line"><span class=pl-c># This file is generated automatically through:</span></td>
      </tr>
      <tr>
        <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
        <td id="LC2" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#    d2lbook build lib</span></td>
      </tr>
      <tr>
        <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
        <td id="LC3" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Don&#39;t edit it directly</span></td>
      </tr>
      <tr>
        <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
        <td id="LC4" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
        <td id="LC5" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_preface/index.md</span></td>
      </tr>
      <tr>
        <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
        <td id="LC6" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>collections</span></td>
      </tr>
      <tr>
        <td id="L7" class="blob-num js-line-number" data-line-number="7"></td>
        <td id="LC7" class="blob-code blob-code-inner js-file-line"><span class=pl-k>from</span> <span class=pl-s1>collections</span> <span class=pl-k>import</span> <span class=pl-s1>defaultdict</span></td>
      </tr>
      <tr>
        <td id="L8" class="blob-num js-line-number" data-line-number="8"></td>
        <td id="LC8" class="blob-code blob-code-inner js-file-line"><span class=pl-k>from</span> <span class=pl-v>IPython</span> <span class=pl-k>import</span> <span class=pl-s1>display</span></td>
      </tr>
      <tr>
        <td id="L9" class="blob-num js-line-number" data-line-number="9"></td>
        <td id="LC9" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>math</span></td>
      </tr>
      <tr>
        <td id="L10" class="blob-num js-line-number" data-line-number="10"></td>
        <td id="LC10" class="blob-code blob-code-inner js-file-line"><span class=pl-k>from</span> <span class=pl-s1>matplotlib</span> <span class=pl-k>import</span> <span class=pl-s1>pyplot</span> <span class=pl-k>as</span> <span class=pl-s1>plt</span></td>
      </tr>
      <tr>
        <td id="L11" class="blob-num js-line-number" data-line-number="11"></td>
        <td id="LC11" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>os</span></td>
      </tr>
      <tr>
        <td id="L12" class="blob-num js-line-number" data-line-number="12"></td>
        <td id="LC12" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>pandas</span> <span class=pl-k>as</span> <span class=pl-s1>pd</span></td>
      </tr>
      <tr>
        <td id="L13" class="blob-num js-line-number" data-line-number="13"></td>
        <td id="LC13" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>random</span></td>
      </tr>
      <tr>
        <td id="L14" class="blob-num js-line-number" data-line-number="14"></td>
        <td id="LC14" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>re</span></td>
      </tr>
      <tr>
        <td id="L15" class="blob-num js-line-number" data-line-number="15"></td>
        <td id="LC15" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>shutil</span></td>
      </tr>
      <tr>
        <td id="L16" class="blob-num js-line-number" data-line-number="16"></td>
        <td id="LC16" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>sys</span></td>
      </tr>
      <tr>
        <td id="L17" class="blob-num js-line-number" data-line-number="17"></td>
        <td id="LC17" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>tarfile</span></td>
      </tr>
      <tr>
        <td id="L18" class="blob-num js-line-number" data-line-number="18"></td>
        <td id="LC18" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>time</span></td>
      </tr>
      <tr>
        <td id="L19" class="blob-num js-line-number" data-line-number="19"></td>
        <td id="LC19" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>requests</span></td>
      </tr>
      <tr>
        <td id="L20" class="blob-num js-line-number" data-line-number="20"></td>
        <td id="LC20" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>zipfile</span></td>
      </tr>
      <tr>
        <td id="L21" class="blob-num js-line-number" data-line-number="21"></td>
        <td id="LC21" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>hashlib</span></td>
      </tr>
      <tr>
        <td id="L22" class="blob-num js-line-number" data-line-number="22"></td>
        <td id="LC22" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>d2l</span> <span class=pl-c1>=</span> <span class=pl-s1>sys</span>.<span class=pl-s1>modules</span>[<span class=pl-s1>__name__</span>]</td>
      </tr>
      <tr>
        <td id="L23" class="blob-num js-line-number" data-line-number="23"></td>
        <td id="LC23" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L24" class="blob-num js-line-number" data-line-number="24"></td>
        <td id="LC24" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L25" class="blob-num js-line-number" data-line-number="25"></td>
        <td id="LC25" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_preface/index.md</span></td>
      </tr>
      <tr>
        <td id="L26" class="blob-num js-line-number" data-line-number="26"></td>
        <td id="LC26" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>numpy</span> <span class=pl-k>as</span> <span class=pl-s1>np</span></td>
      </tr>
      <tr>
        <td id="L27" class="blob-num js-line-number" data-line-number="27"></td>
        <td id="LC27" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>torch</span></td>
      </tr>
      <tr>
        <td id="L28" class="blob-num js-line-number" data-line-number="28"></td>
        <td id="LC28" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>torchvision</span></td>
      </tr>
      <tr>
        <td id="L29" class="blob-num js-line-number" data-line-number="29"></td>
        <td id="LC29" class="blob-code blob-code-inner js-file-line"><span class=pl-k>from</span> <span class=pl-s1>torch</span> <span class=pl-k>import</span> <span class=pl-s1>nn</span></td>
      </tr>
      <tr>
        <td id="L30" class="blob-num js-line-number" data-line-number="30"></td>
        <td id="LC30" class="blob-code blob-code-inner js-file-line"><span class=pl-k>from</span> <span class=pl-s1>torch</span>.<span class=pl-s1>nn</span> <span class=pl-k>import</span> <span class=pl-s1>functional</span> <span class=pl-k>as</span> <span class=pl-v>F</span></td>
      </tr>
      <tr>
        <td id="L31" class="blob-num js-line-number" data-line-number="31"></td>
        <td id="LC31" class="blob-code blob-code-inner js-file-line"><span class=pl-k>from</span> <span class=pl-s1>torch</span>.<span class=pl-s1>utils</span> <span class=pl-k>import</span> <span class=pl-s1>data</span></td>
      </tr>
      <tr>
        <td id="L32" class="blob-num js-line-number" data-line-number="32"></td>
        <td id="LC32" class="blob-code blob-code-inner js-file-line"><span class=pl-k>from</span> <span class=pl-s1>torchvision</span> <span class=pl-k>import</span> <span class=pl-s1>transforms</span></td>
      </tr>
      <tr>
        <td id="L33" class="blob-num js-line-number" data-line-number="33"></td>
        <td id="LC33" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L34" class="blob-num js-line-number" data-line-number="34"></td>
        <td id="LC34" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L35" class="blob-num js-line-number" data-line-number="35"></td>
        <td id="LC35" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_preliminaries/pandas.md</span></td>
      </tr>
      <tr>
        <td id="L36" class="blob-num js-line-number" data-line-number="36"></td>
        <td id="LC36" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>mkdir_if_not_exist</span>(<span class=pl-s1>path</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L37" class="blob-num js-line-number" data-line-number="37"></td>
        <td id="LC37" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Make a directory if it does not exist.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L38" class="blob-num js-line-number" data-line-number="38"></td>
        <td id="LC38" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-c1>not</span> <span class=pl-en>isinstance</span>(<span class=pl-s1>path</span>, <span class=pl-s1>str</span>):</td>
      </tr>
      <tr>
        <td id="L39" class="blob-num js-line-number" data-line-number="39"></td>
        <td id="LC39" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>path</span> <span class=pl-c1>=</span> <span class=pl-s1>os</span>.<span class=pl-s1>path</span>.<span class=pl-en>join</span>(<span class=pl-c1>*</span><span class=pl-s1>path</span>)</td>
      </tr>
      <tr>
        <td id="L40" class="blob-num js-line-number" data-line-number="40"></td>
        <td id="LC40" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-c1>not</span> <span class=pl-s1>os</span>.<span class=pl-s1>path</span>.<span class=pl-en>exists</span>(<span class=pl-s1>path</span>):</td>
      </tr>
      <tr>
        <td id="L41" class="blob-num js-line-number" data-line-number="41"></td>
        <td id="LC41" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>os</span>.<span class=pl-en>makedirs</span>(<span class=pl-s1>path</span>)</td>
      </tr>
      <tr>
        <td id="L42" class="blob-num js-line-number" data-line-number="42"></td>
        <td id="LC42" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L43" class="blob-num js-line-number" data-line-number="43"></td>
        <td id="LC43" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L44" class="blob-num js-line-number" data-line-number="44"></td>
        <td id="LC44" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_preliminaries/calculus.md</span></td>
      </tr>
      <tr>
        <td id="L45" class="blob-num js-line-number" data-line-number="45"></td>
        <td id="LC45" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>use_svg_display</span>():  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L46" class="blob-num js-line-number" data-line-number="46"></td>
        <td id="LC46" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Use the svg format to display a plot in Jupyter.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L47" class="blob-num js-line-number" data-line-number="47"></td>
        <td id="LC47" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>display</span>.<span class=pl-en>set_matplotlib_formats</span>(<span class=pl-s>&#39;svg&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L48" class="blob-num js-line-number" data-line-number="48"></td>
        <td id="LC48" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L49" class="blob-num js-line-number" data-line-number="49"></td>
        <td id="LC49" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L50" class="blob-num js-line-number" data-line-number="50"></td>
        <td id="LC50" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_preliminaries/calculus.md</span></td>
      </tr>
      <tr>
        <td id="L51" class="blob-num js-line-number" data-line-number="51"></td>
        <td id="LC51" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>set_figsize</span>(<span class=pl-s1>figsize</span><span class=pl-c1>=</span>(<span class=pl-c1>3.5</span>, <span class=pl-c1>2.5</span>)):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L52" class="blob-num js-line-number" data-line-number="52"></td>
        <td id="LC52" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Set the figure size for matplotlib.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L53" class="blob-num js-line-number" data-line-number="53"></td>
        <td id="LC53" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>use_svg_display</span>()</td>
      </tr>
      <tr>
        <td id="L54" class="blob-num js-line-number" data-line-number="54"></td>
        <td id="LC54" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>d2l</span>.<span class=pl-s1>plt</span>.<span class=pl-s1>rcParams</span>[<span class=pl-s>&#39;figure.figsize&#39;</span>] <span class=pl-c1>=</span> <span class=pl-s1>figsize</span></td>
      </tr>
      <tr>
        <td id="L55" class="blob-num js-line-number" data-line-number="55"></td>
        <td id="LC55" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L56" class="blob-num js-line-number" data-line-number="56"></td>
        <td id="LC56" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L57" class="blob-num js-line-number" data-line-number="57"></td>
        <td id="LC57" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_preliminaries/calculus.md</span></td>
      </tr>
      <tr>
        <td id="L58" class="blob-num js-line-number" data-line-number="58"></td>
        <td id="LC58" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>set_axes</span>(<span class=pl-s1>axes</span>, <span class=pl-s1>xlabel</span>, <span class=pl-s1>ylabel</span>, <span class=pl-s1>xlim</span>, <span class=pl-s1>ylim</span>, <span class=pl-s1>xscale</span>, <span class=pl-s1>yscale</span>, <span class=pl-s1>legend</span>):</td>
      </tr>
      <tr>
        <td id="L59" class="blob-num js-line-number" data-line-number="59"></td>
        <td id="LC59" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Set the axes for matplotlib.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L60" class="blob-num js-line-number" data-line-number="60"></td>
        <td id="LC60" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>axes</span>.<span class=pl-en>set_xlabel</span>(<span class=pl-s1>xlabel</span>)</td>
      </tr>
      <tr>
        <td id="L61" class="blob-num js-line-number" data-line-number="61"></td>
        <td id="LC61" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>axes</span>.<span class=pl-en>set_ylabel</span>(<span class=pl-s1>ylabel</span>)</td>
      </tr>
      <tr>
        <td id="L62" class="blob-num js-line-number" data-line-number="62"></td>
        <td id="LC62" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>axes</span>.<span class=pl-en>set_xscale</span>(<span class=pl-s1>xscale</span>)</td>
      </tr>
      <tr>
        <td id="L63" class="blob-num js-line-number" data-line-number="63"></td>
        <td id="LC63" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>axes</span>.<span class=pl-en>set_yscale</span>(<span class=pl-s1>yscale</span>)</td>
      </tr>
      <tr>
        <td id="L64" class="blob-num js-line-number" data-line-number="64"></td>
        <td id="LC64" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>axes</span>.<span class=pl-en>set_xlim</span>(<span class=pl-s1>xlim</span>)</td>
      </tr>
      <tr>
        <td id="L65" class="blob-num js-line-number" data-line-number="65"></td>
        <td id="LC65" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>axes</span>.<span class=pl-en>set_ylim</span>(<span class=pl-s1>ylim</span>)</td>
      </tr>
      <tr>
        <td id="L66" class="blob-num js-line-number" data-line-number="66"></td>
        <td id="LC66" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-s1>legend</span>:</td>
      </tr>
      <tr>
        <td id="L67" class="blob-num js-line-number" data-line-number="67"></td>
        <td id="LC67" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>axes</span>.<span class=pl-en>legend</span>(<span class=pl-s1>legend</span>)</td>
      </tr>
      <tr>
        <td id="L68" class="blob-num js-line-number" data-line-number="68"></td>
        <td id="LC68" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>axes</span>.<span class=pl-en>grid</span>()</td>
      </tr>
      <tr>
        <td id="L69" class="blob-num js-line-number" data-line-number="69"></td>
        <td id="LC69" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L70" class="blob-num js-line-number" data-line-number="70"></td>
        <td id="LC70" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L71" class="blob-num js-line-number" data-line-number="71"></td>
        <td id="LC71" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_preliminaries/calculus.md</span></td>
      </tr>
      <tr>
        <td id="L72" class="blob-num js-line-number" data-line-number="72"></td>
        <td id="LC72" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>plot</span>(<span class=pl-v>X</span>, <span class=pl-v>Y</span><span class=pl-c1>=</span><span class=pl-c1>None</span>, <span class=pl-s1>xlabel</span><span class=pl-c1>=</span><span class=pl-c1>None</span>, <span class=pl-s1>ylabel</span><span class=pl-c1>=</span><span class=pl-c1>None</span>, <span class=pl-s1>legend</span><span class=pl-c1>=</span><span class=pl-c1>None</span>, <span class=pl-s1>xlim</span><span class=pl-c1>=</span><span class=pl-c1>None</span>,</td>
      </tr>
      <tr>
        <td id="L73" class="blob-num js-line-number" data-line-number="73"></td>
        <td id="LC73" class="blob-code blob-code-inner js-file-line">         <span class=pl-s1>ylim</span><span class=pl-c1>=</span><span class=pl-c1>None</span>, <span class=pl-s1>xscale</span><span class=pl-c1>=</span><span class=pl-s>&#39;linear&#39;</span>, <span class=pl-s1>yscale</span><span class=pl-c1>=</span><span class=pl-s>&#39;linear&#39;</span>,</td>
      </tr>
      <tr>
        <td id="L74" class="blob-num js-line-number" data-line-number="74"></td>
        <td id="LC74" class="blob-code blob-code-inner js-file-line">         <span class=pl-s1>fmts</span><span class=pl-c1>=</span>(<span class=pl-s>&#39;-&#39;</span>, <span class=pl-s>&#39;m--&#39;</span>, <span class=pl-s>&#39;g-.&#39;</span>, <span class=pl-s>&#39;r:&#39;</span>), <span class=pl-s1>figsize</span><span class=pl-c1>=</span>(<span class=pl-c1>3.5</span>, <span class=pl-c1>2.5</span>), <span class=pl-s1>axes</span><span class=pl-c1>=</span><span class=pl-c1>None</span>):</td>
      </tr>
      <tr>
        <td id="L75" class="blob-num js-line-number" data-line-number="75"></td>
        <td id="LC75" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Plot data points.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L76" class="blob-num js-line-number" data-line-number="76"></td>
        <td id="LC76" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-s1>legend</span> <span class=pl-c1>is</span> <span class=pl-c1>None</span>:</td>
      </tr>
      <tr>
        <td id="L77" class="blob-num js-line-number" data-line-number="77"></td>
        <td id="LC77" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>legend</span> <span class=pl-c1>=</span> []</td>
      </tr>
      <tr>
        <td id="L78" class="blob-num js-line-number" data-line-number="78"></td>
        <td id="LC78" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L79" class="blob-num js-line-number" data-line-number="79"></td>
        <td id="LC79" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>set_figsize</span>(<span class=pl-s1>figsize</span>)</td>
      </tr>
      <tr>
        <td id="L80" class="blob-num js-line-number" data-line-number="80"></td>
        <td id="LC80" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>axes</span> <span class=pl-c1>=</span> <span class=pl-s1>axes</span> <span class=pl-k>if</span> <span class=pl-s1>axes</span> <span class=pl-k>else</span> <span class=pl-s1>d2l</span>.<span class=pl-s1>plt</span>.<span class=pl-en>gca</span>()</td>
      </tr>
      <tr>
        <td id="L81" class="blob-num js-line-number" data-line-number="81"></td>
        <td id="LC81" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L82" class="blob-num js-line-number" data-line-number="82"></td>
        <td id="LC82" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Return True if `X` (tensor or list) has 1 axis</span></td>
      </tr>
      <tr>
        <td id="L83" class="blob-num js-line-number" data-line-number="83"></td>
        <td id="LC83" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>has_one_axis</span>(<span class=pl-v>X</span>):</td>
      </tr>
      <tr>
        <td id="L84" class="blob-num js-line-number" data-line-number="84"></td>
        <td id="LC84" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> (<span class=pl-en>hasattr</span>(<span class=pl-v>X</span>, <span class=pl-s>&quot;ndim&quot;</span>) <span class=pl-c1>and</span> <span class=pl-v>X</span>.<span class=pl-s1>ndim</span> <span class=pl-c1>==</span> <span class=pl-c1>1</span> <span class=pl-c1>or</span> <span class=pl-en>isinstance</span>(<span class=pl-v>X</span>, <span class=pl-s1>list</span>)</td>
      </tr>
      <tr>
        <td id="L85" class="blob-num js-line-number" data-line-number="85"></td>
        <td id="LC85" class="blob-code blob-code-inner js-file-line">                <span class=pl-c1>and</span> <span class=pl-c1>not</span> <span class=pl-en>hasattr</span>(<span class=pl-v>X</span>[<span class=pl-c1>0</span>], <span class=pl-s>&quot;__len__&quot;</span>))</td>
      </tr>
      <tr>
        <td id="L86" class="blob-num js-line-number" data-line-number="86"></td>
        <td id="LC86" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L87" class="blob-num js-line-number" data-line-number="87"></td>
        <td id="LC87" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-en>has_one_axis</span>(<span class=pl-v>X</span>):</td>
      </tr>
      <tr>
        <td id="L88" class="blob-num js-line-number" data-line-number="88"></td>
        <td id="LC88" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>X</span> <span class=pl-c1>=</span> [<span class=pl-v>X</span>]</td>
      </tr>
      <tr>
        <td id="L89" class="blob-num js-line-number" data-line-number="89"></td>
        <td id="LC89" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-v>Y</span> <span class=pl-c1>is</span> <span class=pl-c1>None</span>:</td>
      </tr>
      <tr>
        <td id="L90" class="blob-num js-line-number" data-line-number="90"></td>
        <td id="LC90" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>X</span>, <span class=pl-v>Y</span> <span class=pl-c1>=</span> [[]] <span class=pl-c1>*</span> <span class=pl-en>len</span>(<span class=pl-v>X</span>), <span class=pl-v>X</span></td>
      </tr>
      <tr>
        <td id="L91" class="blob-num js-line-number" data-line-number="91"></td>
        <td id="LC91" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>elif</span> <span class=pl-en>has_one_axis</span>(<span class=pl-v>Y</span>):</td>
      </tr>
      <tr>
        <td id="L92" class="blob-num js-line-number" data-line-number="92"></td>
        <td id="LC92" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>Y</span> <span class=pl-c1>=</span> [<span class=pl-v>Y</span>]</td>
      </tr>
      <tr>
        <td id="L93" class="blob-num js-line-number" data-line-number="93"></td>
        <td id="LC93" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-en>len</span>(<span class=pl-v>X</span>) <span class=pl-c1>!=</span> <span class=pl-en>len</span>(<span class=pl-v>Y</span>):</td>
      </tr>
      <tr>
        <td id="L94" class="blob-num js-line-number" data-line-number="94"></td>
        <td id="LC94" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>X</span> <span class=pl-c1>=</span> <span class=pl-v>X</span> <span class=pl-c1>*</span> <span class=pl-en>len</span>(<span class=pl-v>Y</span>)</td>
      </tr>
      <tr>
        <td id="L95" class="blob-num js-line-number" data-line-number="95"></td>
        <td id="LC95" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>axes</span>.<span class=pl-en>cla</span>()</td>
      </tr>
      <tr>
        <td id="L96" class="blob-num js-line-number" data-line-number="96"></td>
        <td id="LC96" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>x</span>, <span class=pl-s1>y</span>, <span class=pl-s1>fmt</span> <span class=pl-c1>in</span> <span class=pl-en>zip</span>(<span class=pl-v>X</span>, <span class=pl-v>Y</span>, <span class=pl-s1>fmts</span>):</td>
      </tr>
      <tr>
        <td id="L97" class="blob-num js-line-number" data-line-number="97"></td>
        <td id="LC97" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-en>len</span>(<span class=pl-s1>x</span>):</td>
      </tr>
      <tr>
        <td id="L98" class="blob-num js-line-number" data-line-number="98"></td>
        <td id="LC98" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>axes</span>.<span class=pl-en>plot</span>(<span class=pl-s1>x</span>, <span class=pl-s1>y</span>, <span class=pl-s1>fmt</span>)</td>
      </tr>
      <tr>
        <td id="L99" class="blob-num js-line-number" data-line-number="99"></td>
        <td id="LC99" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>else</span>:</td>
      </tr>
      <tr>
        <td id="L100" class="blob-num js-line-number" data-line-number="100"></td>
        <td id="LC100" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>axes</span>.<span class=pl-en>plot</span>(<span class=pl-s1>y</span>, <span class=pl-s1>fmt</span>)</td>
      </tr>
      <tr>
        <td id="L101" class="blob-num js-line-number" data-line-number="101"></td>
        <td id="LC101" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>set_axes</span>(<span class=pl-s1>axes</span>, <span class=pl-s1>xlabel</span>, <span class=pl-s1>ylabel</span>, <span class=pl-s1>xlim</span>, <span class=pl-s1>ylim</span>, <span class=pl-s1>xscale</span>, <span class=pl-s1>yscale</span>, <span class=pl-s1>legend</span>)</td>
      </tr>
      <tr>
        <td id="L102" class="blob-num js-line-number" data-line-number="102"></td>
        <td id="LC102" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L103" class="blob-num js-line-number" data-line-number="103"></td>
        <td id="LC103" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L104" class="blob-num js-line-number" data-line-number="104"></td>
        <td id="LC104" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/linear-regression.md</span></td>
      </tr>
      <tr>
        <td id="L105" class="blob-num js-line-number" data-line-number="105"></td>
        <td id="LC105" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>Timer</span>:  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L106" class="blob-num js-line-number" data-line-number="106"></td>
        <td id="LC106" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Record multiple running times.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L107" class="blob-num js-line-number" data-line-number="107"></td>
        <td id="LC107" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>):</td>
      </tr>
      <tr>
        <td id="L108" class="blob-num js-line-number" data-line-number="108"></td>
        <td id="LC108" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>times</span> <span class=pl-c1>=</span> []</td>
      </tr>
      <tr>
        <td id="L109" class="blob-num js-line-number" data-line-number="109"></td>
        <td id="LC109" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-en>start</span>()</td>
      </tr>
      <tr>
        <td id="L110" class="blob-num js-line-number" data-line-number="110"></td>
        <td id="LC110" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L111" class="blob-num js-line-number" data-line-number="111"></td>
        <td id="LC111" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>start</span>(<span class=pl-s1>self</span>):</td>
      </tr>
      <tr>
        <td id="L112" class="blob-num js-line-number" data-line-number="112"></td>
        <td id="LC112" class="blob-code blob-code-inner js-file-line">        <span class=pl-s>&quot;&quot;&quot;Start the timer.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L113" class="blob-num js-line-number" data-line-number="113"></td>
        <td id="LC113" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>tik</span> <span class=pl-c1>=</span> <span class=pl-s1>time</span>.<span class=pl-en>time</span>()</td>
      </tr>
      <tr>
        <td id="L114" class="blob-num js-line-number" data-line-number="114"></td>
        <td id="LC114" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L115" class="blob-num js-line-number" data-line-number="115"></td>
        <td id="LC115" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>stop</span>(<span class=pl-s1>self</span>):</td>
      </tr>
      <tr>
        <td id="L116" class="blob-num js-line-number" data-line-number="116"></td>
        <td id="LC116" class="blob-code blob-code-inner js-file-line">        <span class=pl-s>&quot;&quot;&quot;Stop the timer and record the time in a list.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L117" class="blob-num js-line-number" data-line-number="117"></td>
        <td id="LC117" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>times</span>.<span class=pl-en>append</span>(<span class=pl-s1>time</span>.<span class=pl-en>time</span>() <span class=pl-c1>-</span> <span class=pl-s1>self</span>.<span class=pl-s1>tik</span>)</td>
      </tr>
      <tr>
        <td id="L118" class="blob-num js-line-number" data-line-number="118"></td>
        <td id="LC118" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>self</span>.<span class=pl-s1>times</span>[<span class=pl-c1>-</span><span class=pl-c1>1</span>]</td>
      </tr>
      <tr>
        <td id="L119" class="blob-num js-line-number" data-line-number="119"></td>
        <td id="LC119" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L120" class="blob-num js-line-number" data-line-number="120"></td>
        <td id="LC120" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>avg</span>(<span class=pl-s1>self</span>):</td>
      </tr>
      <tr>
        <td id="L121" class="blob-num js-line-number" data-line-number="121"></td>
        <td id="LC121" class="blob-code blob-code-inner js-file-line">        <span class=pl-s>&quot;&quot;&quot;Return the average time.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L122" class="blob-num js-line-number" data-line-number="122"></td>
        <td id="LC122" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-en>sum</span>(<span class=pl-s1>self</span>.<span class=pl-s1>times</span>) <span class=pl-c1>/</span> <span class=pl-en>len</span>(<span class=pl-s1>self</span>.<span class=pl-s1>times</span>)</td>
      </tr>
      <tr>
        <td id="L123" class="blob-num js-line-number" data-line-number="123"></td>
        <td id="LC123" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L124" class="blob-num js-line-number" data-line-number="124"></td>
        <td id="LC124" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>sum</span>(<span class=pl-s1>self</span>):</td>
      </tr>
      <tr>
        <td id="L125" class="blob-num js-line-number" data-line-number="125"></td>
        <td id="LC125" class="blob-code blob-code-inner js-file-line">        <span class=pl-s>&quot;&quot;&quot;Return the sum of time.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L126" class="blob-num js-line-number" data-line-number="126"></td>
        <td id="LC126" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-en>sum</span>(<span class=pl-s1>self</span>.<span class=pl-s1>times</span>)</td>
      </tr>
      <tr>
        <td id="L127" class="blob-num js-line-number" data-line-number="127"></td>
        <td id="LC127" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L128" class="blob-num js-line-number" data-line-number="128"></td>
        <td id="LC128" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>cumsum</span>(<span class=pl-s1>self</span>):</td>
      </tr>
      <tr>
        <td id="L129" class="blob-num js-line-number" data-line-number="129"></td>
        <td id="LC129" class="blob-code blob-code-inner js-file-line">        <span class=pl-s>&quot;&quot;&quot;Return the accumulated time.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L130" class="blob-num js-line-number" data-line-number="130"></td>
        <td id="LC130" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>np</span>.<span class=pl-en>array</span>(<span class=pl-s1>self</span>.<span class=pl-s1>times</span>).<span class=pl-en>cumsum</span>().<span class=pl-en>tolist</span>()</td>
      </tr>
      <tr>
        <td id="L131" class="blob-num js-line-number" data-line-number="131"></td>
        <td id="LC131" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L132" class="blob-num js-line-number" data-line-number="132"></td>
        <td id="LC132" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L133" class="blob-num js-line-number" data-line-number="133"></td>
        <td id="LC133" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/linear-regression-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L134" class="blob-num js-line-number" data-line-number="134"></td>
        <td id="LC134" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>synthetic_data</span>(<span class=pl-s1>w</span>, <span class=pl-s1>b</span>, <span class=pl-s1>num_examples</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L135" class="blob-num js-line-number" data-line-number="135"></td>
        <td id="LC135" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Generate y = Xw + b + noise.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L136" class="blob-num js-line-number" data-line-number="136"></td>
        <td id="LC136" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>X</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>normal</span>(<span class=pl-c1>0</span>, <span class=pl-c1>1</span>, (<span class=pl-s1>num_examples</span>, <span class=pl-en>len</span>(<span class=pl-s1>w</span>)))</td>
      </tr>
      <tr>
        <td id="L137" class="blob-num js-line-number" data-line-number="137"></td>
        <td id="LC137" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>matmul</span>(<span class=pl-v>X</span>, <span class=pl-s1>w</span>) <span class=pl-c1>+</span> <span class=pl-s1>b</span></td>
      </tr>
      <tr>
        <td id="L138" class="blob-num js-line-number" data-line-number="138"></td>
        <td id="LC138" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>y</span> <span class=pl-c1>+=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>normal</span>(<span class=pl-c1>0</span>, <span class=pl-c1>0.01</span>, <span class=pl-s1>y</span>.<span class=pl-s1>shape</span>)</td>
      </tr>
      <tr>
        <td id="L139" class="blob-num js-line-number" data-line-number="139"></td>
        <td id="LC139" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>X</span>, <span class=pl-s1>d2l</span>.<span class=pl-en>reshape</span>(<span class=pl-s1>y</span>, (<span class=pl-c1>-</span><span class=pl-c1>1</span>, <span class=pl-c1>1</span>))</td>
      </tr>
      <tr>
        <td id="L140" class="blob-num js-line-number" data-line-number="140"></td>
        <td id="LC140" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L141" class="blob-num js-line-number" data-line-number="141"></td>
        <td id="LC141" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L142" class="blob-num js-line-number" data-line-number="142"></td>
        <td id="LC142" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/linear-regression-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L143" class="blob-num js-line-number" data-line-number="143"></td>
        <td id="LC143" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>linreg</span>(<span class=pl-v>X</span>, <span class=pl-s1>w</span>, <span class=pl-s1>b</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L144" class="blob-num js-line-number" data-line-number="144"></td>
        <td id="LC144" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;The linear regression model.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L145" class="blob-num js-line-number" data-line-number="145"></td>
        <td id="LC145" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>d2l</span>.<span class=pl-en>matmul</span>(<span class=pl-v>X</span>, <span class=pl-s1>w</span>) <span class=pl-c1>+</span> <span class=pl-s1>b</span></td>
      </tr>
      <tr>
        <td id="L146" class="blob-num js-line-number" data-line-number="146"></td>
        <td id="LC146" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L147" class="blob-num js-line-number" data-line-number="147"></td>
        <td id="LC147" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L148" class="blob-num js-line-number" data-line-number="148"></td>
        <td id="LC148" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/linear-regression-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L149" class="blob-num js-line-number" data-line-number="149"></td>
        <td id="LC149" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>squared_loss</span>(<span class=pl-s1>y_hat</span>, <span class=pl-s1>y</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L150" class="blob-num js-line-number" data-line-number="150"></td>
        <td id="LC150" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Squared loss.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L151" class="blob-num js-line-number" data-line-number="151"></td>
        <td id="LC151" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> (<span class=pl-s1>y_hat</span> <span class=pl-c1>-</span> <span class=pl-s1>d2l</span>.<span class=pl-en>reshape</span>(<span class=pl-s1>y</span>, <span class=pl-s1>y_hat</span>.<span class=pl-s1>shape</span>)) <span class=pl-c1>**</span> <span class=pl-c1>2</span> <span class=pl-c1>/</span> <span class=pl-c1>2</span></td>
      </tr>
      <tr>
        <td id="L152" class="blob-num js-line-number" data-line-number="152"></td>
        <td id="LC152" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L153" class="blob-num js-line-number" data-line-number="153"></td>
        <td id="LC153" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L154" class="blob-num js-line-number" data-line-number="154"></td>
        <td id="LC154" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/linear-regression-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L155" class="blob-num js-line-number" data-line-number="155"></td>
        <td id="LC155" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>sgd</span>(<span class=pl-s1>params</span>, <span class=pl-s1>lr</span>, <span class=pl-s1>batch_size</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L156" class="blob-num js-line-number" data-line-number="156"></td>
        <td id="LC156" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Minibatch stochastic gradient descent.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L157" class="blob-num js-line-number" data-line-number="157"></td>
        <td id="LC157" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>param</span> <span class=pl-c1>in</span> <span class=pl-s1>params</span>:</td>
      </tr>
      <tr>
        <td id="L158" class="blob-num js-line-number" data-line-number="158"></td>
        <td id="LC158" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>param</span>.<span class=pl-s1>data</span>.<span class=pl-en>sub_</span>(<span class=pl-s1>lr</span><span class=pl-c1>*</span><span class=pl-s1>param</span>.<span class=pl-s1>grad</span><span class=pl-c1>/</span><span class=pl-s1>batch_size</span>)</td>
      </tr>
      <tr>
        <td id="L159" class="blob-num js-line-number" data-line-number="159"></td>
        <td id="LC159" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>param</span>.<span class=pl-s1>grad</span>.<span class=pl-s1>data</span>.<span class=pl-en>zero_</span>()</td>
      </tr>
      <tr>
        <td id="L160" class="blob-num js-line-number" data-line-number="160"></td>
        <td id="LC160" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L161" class="blob-num js-line-number" data-line-number="161"></td>
        <td id="LC161" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L162" class="blob-num js-line-number" data-line-number="162"></td>
        <td id="LC162" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/linear-regression-concise.md</span></td>
      </tr>
      <tr>
        <td id="L163" class="blob-num js-line-number" data-line-number="163"></td>
        <td id="LC163" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>load_array</span>(<span class=pl-s1>data_arrays</span>, <span class=pl-s1>batch_size</span>, <span class=pl-s1>is_train</span><span class=pl-c1>=</span><span class=pl-c1>True</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L164" class="blob-num js-line-number" data-line-number="164"></td>
        <td id="LC164" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Construct a PyTorch data iterator.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L165" class="blob-num js-line-number" data-line-number="165"></td>
        <td id="LC165" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>dataset</span> <span class=pl-c1>=</span> <span class=pl-s1>data</span>.<span class=pl-v>TensorDataset</span>(<span class=pl-c1>*</span><span class=pl-s1>data_arrays</span>)</td>
      </tr>
      <tr>
        <td id="L166" class="blob-num js-line-number" data-line-number="166"></td>
        <td id="LC166" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>data</span>.<span class=pl-v>DataLoader</span>(<span class=pl-s1>dataset</span>, <span class=pl-s1>batch_size</span>, <span class=pl-s1>shuffle</span><span class=pl-c1>=</span><span class=pl-s1>is_train</span>)</td>
      </tr>
      <tr>
        <td id="L167" class="blob-num js-line-number" data-line-number="167"></td>
        <td id="LC167" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L168" class="blob-num js-line-number" data-line-number="168"></td>
        <td id="LC168" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L169" class="blob-num js-line-number" data-line-number="169"></td>
        <td id="LC169" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/image-classification-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L170" class="blob-num js-line-number" data-line-number="170"></td>
        <td id="LC170" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>get_fashion_mnist_labels</span>(<span class=pl-s1>labels</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L171" class="blob-num js-line-number" data-line-number="171"></td>
        <td id="LC171" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Return text labels for the Fashion-MNIST dataset.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L172" class="blob-num js-line-number" data-line-number="172"></td>
        <td id="LC172" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>text_labels</span> <span class=pl-c1>=</span> [<span class=pl-s>&#39;t-shirt&#39;</span>, <span class=pl-s>&#39;trouser&#39;</span>, <span class=pl-s>&#39;pullover&#39;</span>, <span class=pl-s>&#39;dress&#39;</span>, <span class=pl-s>&#39;coat&#39;</span>,</td>
      </tr>
      <tr>
        <td id="L173" class="blob-num js-line-number" data-line-number="173"></td>
        <td id="LC173" class="blob-code blob-code-inner js-file-line">                   <span class=pl-s>&#39;sandal&#39;</span>, <span class=pl-s>&#39;shirt&#39;</span>, <span class=pl-s>&#39;sneaker&#39;</span>, <span class=pl-s>&#39;bag&#39;</span>, <span class=pl-s>&#39;ankle boot&#39;</span>]</td>
      </tr>
      <tr>
        <td id="L174" class="blob-num js-line-number" data-line-number="174"></td>
        <td id="LC174" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> [<span class=pl-s1>text_labels</span>[<span class=pl-en>int</span>(<span class=pl-s1>i</span>)] <span class=pl-k>for</span> <span class=pl-s1>i</span> <span class=pl-c1>in</span> <span class=pl-s1>labels</span>]</td>
      </tr>
      <tr>
        <td id="L175" class="blob-num js-line-number" data-line-number="175"></td>
        <td id="LC175" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L176" class="blob-num js-line-number" data-line-number="176"></td>
        <td id="LC176" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L177" class="blob-num js-line-number" data-line-number="177"></td>
        <td id="LC177" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/image-classification-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L178" class="blob-num js-line-number" data-line-number="178"></td>
        <td id="LC178" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>show_images</span>(<span class=pl-s1>imgs</span>, <span class=pl-s1>num_rows</span>, <span class=pl-s1>num_cols</span>, <span class=pl-s1>titles</span><span class=pl-c1>=</span><span class=pl-c1>None</span>, <span class=pl-s1>scale</span><span class=pl-c1>=</span><span class=pl-c1>1.5</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L179" class="blob-num js-line-number" data-line-number="179"></td>
        <td id="LC179" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Plot a list of images.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L180" class="blob-num js-line-number" data-line-number="180"></td>
        <td id="LC180" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>figsize</span> <span class=pl-c1>=</span> (<span class=pl-s1>num_cols</span> <span class=pl-c1>*</span> <span class=pl-s1>scale</span>, <span class=pl-s1>num_rows</span> <span class=pl-c1>*</span> <span class=pl-s1>scale</span>)</td>
      </tr>
      <tr>
        <td id="L181" class="blob-num js-line-number" data-line-number="181"></td>
        <td id="LC181" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>_</span>, <span class=pl-s1>axes</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-s1>plt</span>.<span class=pl-en>subplots</span>(<span class=pl-s1>num_rows</span>, <span class=pl-s1>num_cols</span>, <span class=pl-s1>figsize</span><span class=pl-c1>=</span><span class=pl-s1>figsize</span>)</td>
      </tr>
      <tr>
        <td id="L182" class="blob-num js-line-number" data-line-number="182"></td>
        <td id="LC182" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>axes</span> <span class=pl-c1>=</span> <span class=pl-s1>axes</span>.<span class=pl-en>flatten</span>()</td>
      </tr>
      <tr>
        <td id="L183" class="blob-num js-line-number" data-line-number="183"></td>
        <td id="LC183" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>i</span>, (<span class=pl-s1>ax</span>, <span class=pl-s1>img</span>) <span class=pl-c1>in</span> <span class=pl-en>enumerate</span>(<span class=pl-en>zip</span>(<span class=pl-s1>axes</span>, <span class=pl-s1>imgs</span>)):</td>
      </tr>
      <tr>
        <td id="L184" class="blob-num js-line-number" data-line-number="184"></td>
        <td id="LC184" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>ax</span>.<span class=pl-en>imshow</span>(<span class=pl-s1>d2l</span>.<span class=pl-en>numpy</span>(<span class=pl-s1>img</span>))</td>
      </tr>
      <tr>
        <td id="L185" class="blob-num js-line-number" data-line-number="185"></td>
        <td id="LC185" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>ax</span>.<span class=pl-s1>axes</span>.<span class=pl-en>get_xaxis</span>().<span class=pl-en>set_visible</span>(<span class=pl-c1>False</span>)</td>
      </tr>
      <tr>
        <td id="L186" class="blob-num js-line-number" data-line-number="186"></td>
        <td id="LC186" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>ax</span>.<span class=pl-s1>axes</span>.<span class=pl-en>get_yaxis</span>().<span class=pl-en>set_visible</span>(<span class=pl-c1>False</span>)</td>
      </tr>
      <tr>
        <td id="L187" class="blob-num js-line-number" data-line-number="187"></td>
        <td id="LC187" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>titles</span>:</td>
      </tr>
      <tr>
        <td id="L188" class="blob-num js-line-number" data-line-number="188"></td>
        <td id="LC188" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>ax</span>.<span class=pl-en>set_title</span>(<span class=pl-s1>titles</span>[<span class=pl-s1>i</span>])</td>
      </tr>
      <tr>
        <td id="L189" class="blob-num js-line-number" data-line-number="189"></td>
        <td id="LC189" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>axes</span></td>
      </tr>
      <tr>
        <td id="L190" class="blob-num js-line-number" data-line-number="190"></td>
        <td id="LC190" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L191" class="blob-num js-line-number" data-line-number="191"></td>
        <td id="LC191" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L192" class="blob-num js-line-number" data-line-number="192"></td>
        <td id="LC192" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/image-classification-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L193" class="blob-num js-line-number" data-line-number="193"></td>
        <td id="LC193" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>get_dataloader_workers</span>():  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L194" class="blob-num js-line-number" data-line-number="194"></td>
        <td id="LC194" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Use 4 processes to read the data.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L195" class="blob-num js-line-number" data-line-number="195"></td>
        <td id="LC195" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-c1>4</span></td>
      </tr>
      <tr>
        <td id="L196" class="blob-num js-line-number" data-line-number="196"></td>
        <td id="LC196" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L197" class="blob-num js-line-number" data-line-number="197"></td>
        <td id="LC197" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L198" class="blob-num js-line-number" data-line-number="198"></td>
        <td id="LC198" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/image-classification-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L199" class="blob-num js-line-number" data-line-number="199"></td>
        <td id="LC199" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>load_data_fashion_mnist</span>(<span class=pl-s1>batch_size</span>, <span class=pl-s1>resize</span><span class=pl-c1>=</span><span class=pl-c1>None</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L200" class="blob-num js-line-number" data-line-number="200"></td>
        <td id="LC200" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Download the Fashion-MNIST dataset and then load it into memory.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L201" class="blob-num js-line-number" data-line-number="201"></td>
        <td id="LC201" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>trans</span> <span class=pl-c1>=</span> [<span class=pl-s1>transforms</span>.<span class=pl-v>ToTensor</span>()]</td>
      </tr>
      <tr>
        <td id="L202" class="blob-num js-line-number" data-line-number="202"></td>
        <td id="LC202" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-s1>resize</span>:</td>
      </tr>
      <tr>
        <td id="L203" class="blob-num js-line-number" data-line-number="203"></td>
        <td id="LC203" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>trans</span>.<span class=pl-en>insert</span>(<span class=pl-c1>0</span>, <span class=pl-s1>transforms</span>.<span class=pl-v>Resize</span>(<span class=pl-s1>resize</span>))</td>
      </tr>
      <tr>
        <td id="L204" class="blob-num js-line-number" data-line-number="204"></td>
        <td id="LC204" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>trans</span> <span class=pl-c1>=</span> <span class=pl-s1>transforms</span>.<span class=pl-v>Compose</span>(<span class=pl-s1>trans</span>)</td>
      </tr>
      <tr>
        <td id="L205" class="blob-num js-line-number" data-line-number="205"></td>
        <td id="LC205" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>mnist_train</span> <span class=pl-c1>=</span> <span class=pl-s1>torchvision</span>.<span class=pl-s1>datasets</span>.<span class=pl-v>FashionMNIST</span>(</td>
      </tr>
      <tr>
        <td id="L206" class="blob-num js-line-number" data-line-number="206"></td>
        <td id="LC206" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>root</span><span class=pl-c1>=</span><span class=pl-s>&quot;../data&quot;</span>, <span class=pl-s1>train</span><span class=pl-c1>=</span><span class=pl-c1>True</span>, <span class=pl-s1>transform</span><span class=pl-c1>=</span><span class=pl-s1>trans</span>, <span class=pl-s1>download</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L207" class="blob-num js-line-number" data-line-number="207"></td>
        <td id="LC207" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>mnist_test</span> <span class=pl-c1>=</span> <span class=pl-s1>torchvision</span>.<span class=pl-s1>datasets</span>.<span class=pl-v>FashionMNIST</span>(</td>
      </tr>
      <tr>
        <td id="L208" class="blob-num js-line-number" data-line-number="208"></td>
        <td id="LC208" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>root</span><span class=pl-c1>=</span><span class=pl-s>&quot;../data&quot;</span>, <span class=pl-s1>train</span><span class=pl-c1>=</span><span class=pl-c1>False</span>, <span class=pl-s1>transform</span><span class=pl-c1>=</span><span class=pl-s1>trans</span>, <span class=pl-s1>download</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L209" class="blob-num js-line-number" data-line-number="209"></td>
        <td id="LC209" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> (<span class=pl-s1>data</span>.<span class=pl-v>DataLoader</span>(<span class=pl-s1>mnist_train</span>, <span class=pl-s1>batch_size</span>, <span class=pl-s1>shuffle</span><span class=pl-c1>=</span><span class=pl-c1>True</span>,</td>
      </tr>
      <tr>
        <td id="L210" class="blob-num js-line-number" data-line-number="210"></td>
        <td id="LC210" class="blob-code blob-code-inner js-file-line">                            <span class=pl-s1>num_workers</span><span class=pl-c1>=</span><span class=pl-en>get_dataloader_workers</span>()),</td>
      </tr>
      <tr>
        <td id="L211" class="blob-num js-line-number" data-line-number="211"></td>
        <td id="LC211" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>data</span>.<span class=pl-v>DataLoader</span>(<span class=pl-s1>mnist_test</span>, <span class=pl-s1>batch_size</span>, <span class=pl-s1>shuffle</span><span class=pl-c1>=</span><span class=pl-c1>False</span>,</td>
      </tr>
      <tr>
        <td id="L212" class="blob-num js-line-number" data-line-number="212"></td>
        <td id="LC212" class="blob-code blob-code-inner js-file-line">                            <span class=pl-s1>num_workers</span><span class=pl-c1>=</span><span class=pl-en>get_dataloader_workers</span>()))</td>
      </tr>
      <tr>
        <td id="L213" class="blob-num js-line-number" data-line-number="213"></td>
        <td id="LC213" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L214" class="blob-num js-line-number" data-line-number="214"></td>
        <td id="LC214" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L215" class="blob-num js-line-number" data-line-number="215"></td>
        <td id="LC215" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L216" class="blob-num js-line-number" data-line-number="216"></td>
        <td id="LC216" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>accuracy</span>(<span class=pl-s1>y_hat</span>, <span class=pl-s1>y</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L217" class="blob-num js-line-number" data-line-number="217"></td>
        <td id="LC217" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Compute the number of correct predictions.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L218" class="blob-num js-line-number" data-line-number="218"></td>
        <td id="LC218" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-en>len</span>(<span class=pl-s1>y_hat</span>.<span class=pl-s1>shape</span>) <span class=pl-c1>&gt;</span> <span class=pl-c1>1</span> <span class=pl-c1>and</span> <span class=pl-s1>y_hat</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>1</span>] <span class=pl-c1>&gt;</span> <span class=pl-c1>1</span>:</td>
      </tr>
      <tr>
        <td id="L219" class="blob-num js-line-number" data-line-number="219"></td>
        <td id="LC219" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>y_hat</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>argmax</span>(<span class=pl-s1>y_hat</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>)        </td>
      </tr>
      <tr>
        <td id="L220" class="blob-num js-line-number" data-line-number="220"></td>
        <td id="LC220" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>cmp</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>astype</span>(<span class=pl-s1>y_hat</span>, <span class=pl-s1>y</span>.<span class=pl-s1>dtype</span>) <span class=pl-c1>==</span> <span class=pl-s1>y</span></td>
      </tr>
      <tr>
        <td id="L221" class="blob-num js-line-number" data-line-number="221"></td>
        <td id="LC221" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-en>float</span>(<span class=pl-s1>d2l</span>.<span class=pl-en>reduce_sum</span>(<span class=pl-s1>d2l</span>.<span class=pl-en>astype</span>(<span class=pl-s1>cmp</span>, <span class=pl-s1>y</span>.<span class=pl-s1>dtype</span>)))</td>
      </tr>
      <tr>
        <td id="L222" class="blob-num js-line-number" data-line-number="222"></td>
        <td id="LC222" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L223" class="blob-num js-line-number" data-line-number="223"></td>
        <td id="LC223" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L224" class="blob-num js-line-number" data-line-number="224"></td>
        <td id="LC224" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L225" class="blob-num js-line-number" data-line-number="225"></td>
        <td id="LC225" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>evaluate_accuracy</span>(<span class=pl-s1>net</span>, <span class=pl-s1>data_iter</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L226" class="blob-num js-line-number" data-line-number="226"></td>
        <td id="LC226" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Compute the accuracy for a model on a dataset.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L227" class="blob-num js-line-number" data-line-number="227"></td>
        <td id="LC227" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-en>isinstance</span>(<span class=pl-s1>net</span>, <span class=pl-s1>torch</span>.<span class=pl-s1>nn</span>.<span class=pl-v>Module</span>):</td>
      </tr>
      <tr>
        <td id="L228" class="blob-num js-line-number" data-line-number="228"></td>
        <td id="LC228" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>net</span>.<span class=pl-en>eval</span>()  <span class=pl-c># Set the model to evaluation mode</span></td>
      </tr>
      <tr>
        <td id="L229" class="blob-num js-line-number" data-line-number="229"></td>
        <td id="LC229" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>metric</span> <span class=pl-c1>=</span> <span class=pl-v>Accumulator</span>(<span class=pl-c1>2</span>)  <span class=pl-c># No. of correct predictions, no. of predictions</span></td>
      </tr>
      <tr>
        <td id="L230" class="blob-num js-line-number" data-line-number="230"></td>
        <td id="LC230" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>_</span>, (<span class=pl-v>X</span>, <span class=pl-s1>y</span>) <span class=pl-c1>in</span> <span class=pl-en>enumerate</span>(<span class=pl-s1>data_iter</span>):</td>
      </tr>
      <tr>
        <td id="L231" class="blob-num js-line-number" data-line-number="231"></td>
        <td id="LC231" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>metric</span>.<span class=pl-en>add</span>(<span class=pl-en>accuracy</span>(<span class=pl-en>net</span>(<span class=pl-v>X</span>), <span class=pl-s1>y</span>), <span class=pl-s1>d2l</span>.<span class=pl-en>size</span>(<span class=pl-s1>y</span>))</td>
      </tr>
      <tr>
        <td id="L232" class="blob-num js-line-number" data-line-number="232"></td>
        <td id="LC232" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>metric</span>[<span class=pl-c1>0</span>] <span class=pl-c1>/</span> <span class=pl-s1>metric</span>[<span class=pl-c1>1</span>]</td>
      </tr>
      <tr>
        <td id="L233" class="blob-num js-line-number" data-line-number="233"></td>
        <td id="LC233" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L234" class="blob-num js-line-number" data-line-number="234"></td>
        <td id="LC234" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L235" class="blob-num js-line-number" data-line-number="235"></td>
        <td id="LC235" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L236" class="blob-num js-line-number" data-line-number="236"></td>
        <td id="LC236" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>Accumulator</span>:  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L237" class="blob-num js-line-number" data-line-number="237"></td>
        <td id="LC237" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;For accumulating sums over `n` variables.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L238" class="blob-num js-line-number" data-line-number="238"></td>
        <td id="LC238" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>, <span class=pl-s1>n</span>):</td>
      </tr>
      <tr>
        <td id="L239" class="blob-num js-line-number" data-line-number="239"></td>
        <td id="LC239" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>data</span> <span class=pl-c1>=</span> [<span class=pl-c1>0.0</span>] <span class=pl-c1>*</span> <span class=pl-s1>n</span></td>
      </tr>
      <tr>
        <td id="L240" class="blob-num js-line-number" data-line-number="240"></td>
        <td id="LC240" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L241" class="blob-num js-line-number" data-line-number="241"></td>
        <td id="LC241" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>add</span>(<span class=pl-s1>self</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>):</td>
      </tr>
      <tr>
        <td id="L242" class="blob-num js-line-number" data-line-number="242"></td>
        <td id="LC242" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>data</span> <span class=pl-c1>=</span> [<span class=pl-s1>a</span> <span class=pl-c1>+</span> <span class=pl-en>float</span>(<span class=pl-s1>b</span>) <span class=pl-k>for</span> <span class=pl-s1>a</span>, <span class=pl-s1>b</span> <span class=pl-c1>in</span> <span class=pl-en>zip</span>(<span class=pl-s1>self</span>.<span class=pl-s1>data</span>, <span class=pl-s1>args</span>)]</td>
      </tr>
      <tr>
        <td id="L243" class="blob-num js-line-number" data-line-number="243"></td>
        <td id="LC243" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L244" class="blob-num js-line-number" data-line-number="244"></td>
        <td id="LC244" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>reset</span>(<span class=pl-s1>self</span>):</td>
      </tr>
      <tr>
        <td id="L245" class="blob-num js-line-number" data-line-number="245"></td>
        <td id="LC245" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>data</span> <span class=pl-c1>=</span> [<span class=pl-c1>0.0</span>] <span class=pl-c1>*</span> <span class=pl-en>len</span>(<span class=pl-s1>self</span>.<span class=pl-s1>data</span>)</td>
      </tr>
      <tr>
        <td id="L246" class="blob-num js-line-number" data-line-number="246"></td>
        <td id="LC246" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L247" class="blob-num js-line-number" data-line-number="247"></td>
        <td id="LC247" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__getitem__</span>(<span class=pl-s1>self</span>, <span class=pl-s1>idx</span>):</td>
      </tr>
      <tr>
        <td id="L248" class="blob-num js-line-number" data-line-number="248"></td>
        <td id="LC248" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>self</span>.<span class=pl-s1>data</span>[<span class=pl-s1>idx</span>]</td>
      </tr>
      <tr>
        <td id="L249" class="blob-num js-line-number" data-line-number="249"></td>
        <td id="LC249" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L250" class="blob-num js-line-number" data-line-number="250"></td>
        <td id="LC250" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L251" class="blob-num js-line-number" data-line-number="251"></td>
        <td id="LC251" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L252" class="blob-num js-line-number" data-line-number="252"></td>
        <td id="LC252" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>train_epoch_ch3</span>(<span class=pl-s1>net</span>, <span class=pl-s1>train_iter</span>, <span class=pl-s1>loss</span>, <span class=pl-s1>updater</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L253" class="blob-num js-line-number" data-line-number="253"></td>
        <td id="LC253" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;The training loop defined in Chapter 3.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L254" class="blob-num js-line-number" data-line-number="254"></td>
        <td id="LC254" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Set the model to training mode</span></td>
      </tr>
      <tr>
        <td id="L255" class="blob-num js-line-number" data-line-number="255"></td>
        <td id="LC255" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-en>isinstance</span>(<span class=pl-s1>net</span>, <span class=pl-s1>torch</span>.<span class=pl-s1>nn</span>.<span class=pl-v>Module</span>):</td>
      </tr>
      <tr>
        <td id="L256" class="blob-num js-line-number" data-line-number="256"></td>
        <td id="LC256" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>net</span>.<span class=pl-en>train</span>()</td>
      </tr>
      <tr>
        <td id="L257" class="blob-num js-line-number" data-line-number="257"></td>
        <td id="LC257" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Sum of training loss, sum of training accuracy, no. of examples</span></td>
      </tr>
      <tr>
        <td id="L258" class="blob-num js-line-number" data-line-number="258"></td>
        <td id="LC258" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>metric</span> <span class=pl-c1>=</span> <span class=pl-v>Accumulator</span>(<span class=pl-c1>3</span>)</td>
      </tr>
      <tr>
        <td id="L259" class="blob-num js-line-number" data-line-number="259"></td>
        <td id="LC259" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-v>X</span>, <span class=pl-s1>y</span> <span class=pl-c1>in</span> <span class=pl-s1>train_iter</span>:</td>
      </tr>
      <tr>
        <td id="L260" class="blob-num js-line-number" data-line-number="260"></td>
        <td id="LC260" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># Compute gradients and update parameters</span></td>
      </tr>
      <tr>
        <td id="L261" class="blob-num js-line-number" data-line-number="261"></td>
        <td id="LC261" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>y_hat</span> <span class=pl-c1>=</span> <span class=pl-en>net</span>(<span class=pl-v>X</span>)</td>
      </tr>
      <tr>
        <td id="L262" class="blob-num js-line-number" data-line-number="262"></td>
        <td id="LC262" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>l</span> <span class=pl-c1>=</span> <span class=pl-en>loss</span>(<span class=pl-s1>y_hat</span>, <span class=pl-s1>y</span>)</td>
      </tr>
      <tr>
        <td id="L263" class="blob-num js-line-number" data-line-number="263"></td>
        <td id="LC263" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-en>isinstance</span>(<span class=pl-s1>updater</span>, <span class=pl-s1>torch</span>.<span class=pl-s1>optim</span>.<span class=pl-v>Optimizer</span>):</td>
      </tr>
      <tr>
        <td id="L264" class="blob-num js-line-number" data-line-number="264"></td>
        <td id="LC264" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>updater</span>.<span class=pl-en>zero_grad</span>()</td>
      </tr>
      <tr>
        <td id="L265" class="blob-num js-line-number" data-line-number="265"></td>
        <td id="LC265" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>l</span>.<span class=pl-en>backward</span>()</td>
      </tr>
      <tr>
        <td id="L266" class="blob-num js-line-number" data-line-number="266"></td>
        <td id="LC266" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>updater</span>.<span class=pl-en>step</span>()</td>
      </tr>
      <tr>
        <td id="L267" class="blob-num js-line-number" data-line-number="267"></td>
        <td id="LC267" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>metric</span>.<span class=pl-en>add</span>(<span class=pl-en>float</span>(<span class=pl-s1>l</span>) <span class=pl-c1>*</span> <span class=pl-en>len</span>(<span class=pl-s1>y</span>), <span class=pl-en>accuracy</span>(<span class=pl-s1>y_hat</span>, <span class=pl-s1>y</span>),</td>
      </tr>
      <tr>
        <td id="L268" class="blob-num js-line-number" data-line-number="268"></td>
        <td id="LC268" class="blob-code blob-code-inner js-file-line">                       <span class=pl-s1>y</span>.<span class=pl-en>size</span>().<span class=pl-en>numel</span>())</td>
      </tr>
      <tr>
        <td id="L269" class="blob-num js-line-number" data-line-number="269"></td>
        <td id="LC269" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>else</span>:</td>
      </tr>
      <tr>
        <td id="L270" class="blob-num js-line-number" data-line-number="270"></td>
        <td id="LC270" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>l</span>.<span class=pl-en>sum</span>().<span class=pl-en>backward</span>()</td>
      </tr>
      <tr>
        <td id="L271" class="blob-num js-line-number" data-line-number="271"></td>
        <td id="LC271" class="blob-code blob-code-inner js-file-line">            <span class=pl-en>updater</span>(<span class=pl-v>X</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>0</span>])</td>
      </tr>
      <tr>
        <td id="L272" class="blob-num js-line-number" data-line-number="272"></td>
        <td id="LC272" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>metric</span>.<span class=pl-en>add</span>(<span class=pl-en>float</span>(<span class=pl-s1>l</span>.<span class=pl-en>sum</span>()), <span class=pl-en>accuracy</span>(<span class=pl-s1>y_hat</span>, <span class=pl-s1>y</span>), <span class=pl-s1>y</span>.<span class=pl-en>numel</span>())</td>
      </tr>
      <tr>
        <td id="L273" class="blob-num js-line-number" data-line-number="273"></td>
        <td id="LC273" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Return training loss and training accuracy</span></td>
      </tr>
      <tr>
        <td id="L274" class="blob-num js-line-number" data-line-number="274"></td>
        <td id="LC274" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>metric</span>[<span class=pl-c1>0</span>] <span class=pl-c1>/</span> <span class=pl-s1>metric</span>[<span class=pl-c1>2</span>], <span class=pl-s1>metric</span>[<span class=pl-c1>1</span>] <span class=pl-c1>/</span> <span class=pl-s1>metric</span>[<span class=pl-c1>2</span>]</td>
      </tr>
      <tr>
        <td id="L275" class="blob-num js-line-number" data-line-number="275"></td>
        <td id="LC275" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L276" class="blob-num js-line-number" data-line-number="276"></td>
        <td id="LC276" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L277" class="blob-num js-line-number" data-line-number="277"></td>
        <td id="LC277" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L278" class="blob-num js-line-number" data-line-number="278"></td>
        <td id="LC278" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>Animator</span>:  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L279" class="blob-num js-line-number" data-line-number="279"></td>
        <td id="LC279" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;For plotting data in animation.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L280" class="blob-num js-line-number" data-line-number="280"></td>
        <td id="LC280" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>, <span class=pl-s1>xlabel</span><span class=pl-c1>=</span><span class=pl-c1>None</span>, <span class=pl-s1>ylabel</span><span class=pl-c1>=</span><span class=pl-c1>None</span>, <span class=pl-s1>legend</span><span class=pl-c1>=</span><span class=pl-c1>None</span>, <span class=pl-s1>xlim</span><span class=pl-c1>=</span><span class=pl-c1>None</span>,</td>
      </tr>
      <tr>
        <td id="L281" class="blob-num js-line-number" data-line-number="281"></td>
        <td id="LC281" class="blob-code blob-code-inner js-file-line">                 <span class=pl-s1>ylim</span><span class=pl-c1>=</span><span class=pl-c1>None</span>, <span class=pl-s1>xscale</span><span class=pl-c1>=</span><span class=pl-s>&#39;linear&#39;</span>, <span class=pl-s1>yscale</span><span class=pl-c1>=</span><span class=pl-s>&#39;linear&#39;</span>,</td>
      </tr>
      <tr>
        <td id="L282" class="blob-num js-line-number" data-line-number="282"></td>
        <td id="LC282" class="blob-code blob-code-inner js-file-line">                 <span class=pl-s1>fmts</span><span class=pl-c1>=</span>(<span class=pl-s>&#39;-&#39;</span>, <span class=pl-s>&#39;m--&#39;</span>, <span class=pl-s>&#39;g-.&#39;</span>, <span class=pl-s>&#39;r:&#39;</span>), <span class=pl-s1>nrows</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>ncols</span><span class=pl-c1>=</span><span class=pl-c1>1</span>,</td>
      </tr>
      <tr>
        <td id="L283" class="blob-num js-line-number" data-line-number="283"></td>
        <td id="LC283" class="blob-code blob-code-inner js-file-line">                 <span class=pl-s1>figsize</span><span class=pl-c1>=</span>(<span class=pl-c1>3.5</span>, <span class=pl-c1>2.5</span>)):</td>
      </tr>
      <tr>
        <td id="L284" class="blob-num js-line-number" data-line-number="284"></td>
        <td id="LC284" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># Incrementally plot multiple lines</span></td>
      </tr>
      <tr>
        <td id="L285" class="blob-num js-line-number" data-line-number="285"></td>
        <td id="LC285" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>legend</span> <span class=pl-c1>is</span> <span class=pl-c1>None</span>:</td>
      </tr>
      <tr>
        <td id="L286" class="blob-num js-line-number" data-line-number="286"></td>
        <td id="LC286" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>legend</span> <span class=pl-c1>=</span> []</td>
      </tr>
      <tr>
        <td id="L287" class="blob-num js-line-number" data-line-number="287"></td>
        <td id="LC287" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>d2l</span>.<span class=pl-en>use_svg_display</span>()</td>
      </tr>
      <tr>
        <td id="L288" class="blob-num js-line-number" data-line-number="288"></td>
        <td id="LC288" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>fig</span>, <span class=pl-s1>self</span>.<span class=pl-s1>axes</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-s1>plt</span>.<span class=pl-en>subplots</span>(<span class=pl-s1>nrows</span>, <span class=pl-s1>ncols</span>, <span class=pl-s1>figsize</span><span class=pl-c1>=</span><span class=pl-s1>figsize</span>)</td>
      </tr>
      <tr>
        <td id="L289" class="blob-num js-line-number" data-line-number="289"></td>
        <td id="LC289" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>nrows</span> <span class=pl-c1>*</span> <span class=pl-s1>ncols</span> <span class=pl-c1>==</span> <span class=pl-c1>1</span>:</td>
      </tr>
      <tr>
        <td id="L290" class="blob-num js-line-number" data-line-number="290"></td>
        <td id="LC290" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>self</span>.<span class=pl-s1>axes</span> <span class=pl-c1>=</span> [<span class=pl-s1>self</span>.<span class=pl-s1>axes</span>, ]</td>
      </tr>
      <tr>
        <td id="L291" class="blob-num js-line-number" data-line-number="291"></td>
        <td id="LC291" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># Use a lambda function to capture arguments</span></td>
      </tr>
      <tr>
        <td id="L292" class="blob-num js-line-number" data-line-number="292"></td>
        <td id="LC292" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>config_axes</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span>: <span class=pl-s1>d2l</span>.<span class=pl-en>set_axes</span>(</td>
      </tr>
      <tr>
        <td id="L293" class="blob-num js-line-number" data-line-number="293"></td>
        <td id="LC293" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>self</span>.<span class=pl-s1>axes</span>[<span class=pl-c1>0</span>], <span class=pl-s1>xlabel</span>, <span class=pl-s1>ylabel</span>, <span class=pl-s1>xlim</span>, <span class=pl-s1>ylim</span>, <span class=pl-s1>xscale</span>, <span class=pl-s1>yscale</span>, <span class=pl-s1>legend</span>)</td>
      </tr>
      <tr>
        <td id="L294" class="blob-num js-line-number" data-line-number="294"></td>
        <td id="LC294" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-v>X</span>, <span class=pl-s1>self</span>.<span class=pl-v>Y</span>, <span class=pl-s1>self</span>.<span class=pl-s1>fmts</span> <span class=pl-c1>=</span> <span class=pl-c1>None</span>, <span class=pl-c1>None</span>, <span class=pl-s1>fmts</span></td>
      </tr>
      <tr>
        <td id="L295" class="blob-num js-line-number" data-line-number="295"></td>
        <td id="LC295" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L296" class="blob-num js-line-number" data-line-number="296"></td>
        <td id="LC296" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>add</span>(<span class=pl-s1>self</span>, <span class=pl-s1>x</span>, <span class=pl-s1>y</span>):</td>
      </tr>
      <tr>
        <td id="L297" class="blob-num js-line-number" data-line-number="297"></td>
        <td id="LC297" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># Add multiple data points into the figure</span></td>
      </tr>
      <tr>
        <td id="L298" class="blob-num js-line-number" data-line-number="298"></td>
        <td id="LC298" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-c1>not</span> <span class=pl-en>hasattr</span>(<span class=pl-s1>y</span>, <span class=pl-s>&quot;__len__&quot;</span>):</td>
      </tr>
      <tr>
        <td id="L299" class="blob-num js-line-number" data-line-number="299"></td>
        <td id="LC299" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>y</span> <span class=pl-c1>=</span> [<span class=pl-s1>y</span>]</td>
      </tr>
      <tr>
        <td id="L300" class="blob-num js-line-number" data-line-number="300"></td>
        <td id="LC300" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>n</span> <span class=pl-c1>=</span> <span class=pl-en>len</span>(<span class=pl-s1>y</span>)</td>
      </tr>
      <tr>
        <td id="L301" class="blob-num js-line-number" data-line-number="301"></td>
        <td id="LC301" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-c1>not</span> <span class=pl-en>hasattr</span>(<span class=pl-s1>x</span>, <span class=pl-s>&quot;__len__&quot;</span>):</td>
      </tr>
      <tr>
        <td id="L302" class="blob-num js-line-number" data-line-number="302"></td>
        <td id="LC302" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>x</span> <span class=pl-c1>=</span> [<span class=pl-s1>x</span>] <span class=pl-c1>*</span> <span class=pl-s1>n</span></td>
      </tr>
      <tr>
        <td id="L303" class="blob-num js-line-number" data-line-number="303"></td>
        <td id="LC303" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-c1>not</span> <span class=pl-s1>self</span>.<span class=pl-v>X</span>:</td>
      </tr>
      <tr>
        <td id="L304" class="blob-num js-line-number" data-line-number="304"></td>
        <td id="LC304" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>self</span>.<span class=pl-v>X</span> <span class=pl-c1>=</span> [[] <span class=pl-k>for</span> <span class=pl-s1>_</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-s1>n</span>)]</td>
      </tr>
      <tr>
        <td id="L305" class="blob-num js-line-number" data-line-number="305"></td>
        <td id="LC305" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-c1>not</span> <span class=pl-s1>self</span>.<span class=pl-v>Y</span>:</td>
      </tr>
      <tr>
        <td id="L306" class="blob-num js-line-number" data-line-number="306"></td>
        <td id="LC306" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>self</span>.<span class=pl-v>Y</span> <span class=pl-c1>=</span> [[] <span class=pl-k>for</span> <span class=pl-s1>_</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-s1>n</span>)]</td>
      </tr>
      <tr>
        <td id="L307" class="blob-num js-line-number" data-line-number="307"></td>
        <td id="LC307" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-s1>i</span>, (<span class=pl-s1>a</span>, <span class=pl-s1>b</span>) <span class=pl-c1>in</span> <span class=pl-en>enumerate</span>(<span class=pl-en>zip</span>(<span class=pl-s1>x</span>, <span class=pl-s1>y</span>)):</td>
      </tr>
      <tr>
        <td id="L308" class="blob-num js-line-number" data-line-number="308"></td>
        <td id="LC308" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>if</span> <span class=pl-s1>a</span> <span class=pl-c1>is</span> <span class=pl-c1>not</span> <span class=pl-c1>None</span> <span class=pl-c1>and</span> <span class=pl-s1>b</span> <span class=pl-c1>is</span> <span class=pl-c1>not</span> <span class=pl-c1>None</span>:</td>
      </tr>
      <tr>
        <td id="L309" class="blob-num js-line-number" data-line-number="309"></td>
        <td id="LC309" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>self</span>.<span class=pl-v>X</span>[<span class=pl-s1>i</span>].<span class=pl-en>append</span>(<span class=pl-s1>a</span>)</td>
      </tr>
      <tr>
        <td id="L310" class="blob-num js-line-number" data-line-number="310"></td>
        <td id="LC310" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>self</span>.<span class=pl-v>Y</span>[<span class=pl-s1>i</span>].<span class=pl-en>append</span>(<span class=pl-s1>b</span>)</td>
      </tr>
      <tr>
        <td id="L311" class="blob-num js-line-number" data-line-number="311"></td>
        <td id="LC311" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>axes</span>[<span class=pl-c1>0</span>].<span class=pl-en>cla</span>()</td>
      </tr>
      <tr>
        <td id="L312" class="blob-num js-line-number" data-line-number="312"></td>
        <td id="LC312" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-s1>x</span>, <span class=pl-s1>y</span>, <span class=pl-s1>fmt</span> <span class=pl-c1>in</span> <span class=pl-en>zip</span>(<span class=pl-s1>self</span>.<span class=pl-v>X</span>, <span class=pl-s1>self</span>.<span class=pl-v>Y</span>, <span class=pl-s1>self</span>.<span class=pl-s1>fmts</span>):</td>
      </tr>
      <tr>
        <td id="L313" class="blob-num js-line-number" data-line-number="313"></td>
        <td id="LC313" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>self</span>.<span class=pl-s1>axes</span>[<span class=pl-c1>0</span>].<span class=pl-en>plot</span>(<span class=pl-s1>x</span>, <span class=pl-s1>y</span>, <span class=pl-s1>fmt</span>)</td>
      </tr>
      <tr>
        <td id="L314" class="blob-num js-line-number" data-line-number="314"></td>
        <td id="LC314" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-en>config_axes</span>()</td>
      </tr>
      <tr>
        <td id="L315" class="blob-num js-line-number" data-line-number="315"></td>
        <td id="LC315" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>display</span>.<span class=pl-en>display</span>(<span class=pl-s1>self</span>.<span class=pl-s1>fig</span>)</td>
      </tr>
      <tr>
        <td id="L316" class="blob-num js-line-number" data-line-number="316"></td>
        <td id="LC316" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>display</span>.<span class=pl-en>clear_output</span>(<span class=pl-s1>wait</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L317" class="blob-num js-line-number" data-line-number="317"></td>
        <td id="LC317" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L318" class="blob-num js-line-number" data-line-number="318"></td>
        <td id="LC318" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L319" class="blob-num js-line-number" data-line-number="319"></td>
        <td id="LC319" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L320" class="blob-num js-line-number" data-line-number="320"></td>
        <td id="LC320" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>train_ch3</span>(<span class=pl-s1>net</span>, <span class=pl-s1>train_iter</span>, <span class=pl-s1>test_iter</span>, <span class=pl-s1>loss</span>, <span class=pl-s1>num_epochs</span>, <span class=pl-s1>updater</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L321" class="blob-num js-line-number" data-line-number="321"></td>
        <td id="LC321" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Train a model (defined in Chapter 3).&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L322" class="blob-num js-line-number" data-line-number="322"></td>
        <td id="LC322" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>animator</span> <span class=pl-c1>=</span> <span class=pl-v>Animator</span>(<span class=pl-s1>xlabel</span><span class=pl-c1>=</span><span class=pl-s>&#39;epoch&#39;</span>, <span class=pl-s1>xlim</span><span class=pl-c1>=</span>[<span class=pl-c1>1</span>, <span class=pl-s1>num_epochs</span>], <span class=pl-s1>ylim</span><span class=pl-c1>=</span>[<span class=pl-c1>0.3</span>, <span class=pl-c1>0.9</span>],</td>
      </tr>
      <tr>
        <td id="L323" class="blob-num js-line-number" data-line-number="323"></td>
        <td id="LC323" class="blob-code blob-code-inner js-file-line">                        <span class=pl-s1>legend</span><span class=pl-c1>=</span>[<span class=pl-s>&#39;train loss&#39;</span>, <span class=pl-s>&#39;train acc&#39;</span>, <span class=pl-s>&#39;test acc&#39;</span>])</td>
      </tr>
      <tr>
        <td id="L324" class="blob-num js-line-number" data-line-number="324"></td>
        <td id="LC324" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>epoch</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-s1>num_epochs</span>):</td>
      </tr>
      <tr>
        <td id="L325" class="blob-num js-line-number" data-line-number="325"></td>
        <td id="LC325" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>train_metrics</span> <span class=pl-c1>=</span> <span class=pl-en>train_epoch_ch3</span>(<span class=pl-s1>net</span>, <span class=pl-s1>train_iter</span>, <span class=pl-s1>loss</span>, <span class=pl-s1>updater</span>)</td>
      </tr>
      <tr>
        <td id="L326" class="blob-num js-line-number" data-line-number="326"></td>
        <td id="LC326" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>test_acc</span> <span class=pl-c1>=</span> <span class=pl-en>evaluate_accuracy</span>(<span class=pl-s1>net</span>, <span class=pl-s1>test_iter</span>)</td>
      </tr>
      <tr>
        <td id="L327" class="blob-num js-line-number" data-line-number="327"></td>
        <td id="LC327" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>animator</span>.<span class=pl-en>add</span>(<span class=pl-s1>epoch</span> <span class=pl-c1>+</span> <span class=pl-c1>1</span>, <span class=pl-s1>train_metrics</span> <span class=pl-c1>+</span> (<span class=pl-s1>test_acc</span>,))</td>
      </tr>
      <tr>
        <td id="L328" class="blob-num js-line-number" data-line-number="328"></td>
        <td id="LC328" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>train_loss</span>, <span class=pl-s1>train_acc</span> <span class=pl-c1>=</span> <span class=pl-s1>train_metrics</span></td>
      </tr>
      <tr>
        <td id="L329" class="blob-num js-line-number" data-line-number="329"></td>
        <td id="LC329" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>assert</span> <span class=pl-s1>train_loss</span> <span class=pl-c1>&lt;</span> <span class=pl-c1>0.5</span>, <span class=pl-s1>train_loss</span></td>
      </tr>
      <tr>
        <td id="L330" class="blob-num js-line-number" data-line-number="330"></td>
        <td id="LC330" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>assert</span> <span class=pl-s1>train_acc</span> <span class=pl-c1>&lt;=</span> <span class=pl-c1>1</span> <span class=pl-c1>and</span> <span class=pl-s1>train_acc</span> <span class=pl-c1>&gt;</span> <span class=pl-c1>0.7</span>, <span class=pl-s1>train_acc</span></td>
      </tr>
      <tr>
        <td id="L331" class="blob-num js-line-number" data-line-number="331"></td>
        <td id="LC331" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>assert</span> <span class=pl-s1>test_acc</span> <span class=pl-c1>&lt;=</span> <span class=pl-c1>1</span> <span class=pl-c1>and</span> <span class=pl-s1>test_acc</span> <span class=pl-c1>&gt;</span> <span class=pl-c1>0.7</span>, <span class=pl-s1>test_acc</span></td>
      </tr>
      <tr>
        <td id="L332" class="blob-num js-line-number" data-line-number="332"></td>
        <td id="LC332" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L333" class="blob-num js-line-number" data-line-number="333"></td>
        <td id="LC333" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L334" class="blob-num js-line-number" data-line-number="334"></td>
        <td id="LC334" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L335" class="blob-num js-line-number" data-line-number="335"></td>
        <td id="LC335" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>predict_ch3</span>(<span class=pl-s1>net</span>, <span class=pl-s1>test_iter</span>, <span class=pl-s1>n</span><span class=pl-c1>=</span><span class=pl-c1>6</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L336" class="blob-num js-line-number" data-line-number="336"></td>
        <td id="LC336" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Predict labels (defined in Chapter 3).&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L337" class="blob-num js-line-number" data-line-number="337"></td>
        <td id="LC337" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-v>X</span>, <span class=pl-s1>y</span> <span class=pl-c1>in</span> <span class=pl-s1>test_iter</span>:</td>
      </tr>
      <tr>
        <td id="L338" class="blob-num js-line-number" data-line-number="338"></td>
        <td id="LC338" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>break</span></td>
      </tr>
      <tr>
        <td id="L339" class="blob-num js-line-number" data-line-number="339"></td>
        <td id="LC339" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>trues</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>get_fashion_mnist_labels</span>(<span class=pl-s1>y</span>)</td>
      </tr>
      <tr>
        <td id="L340" class="blob-num js-line-number" data-line-number="340"></td>
        <td id="LC340" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>preds</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>get_fashion_mnist_labels</span>(<span class=pl-s1>d2l</span>.<span class=pl-en>argmax</span>(<span class=pl-en>net</span>(<span class=pl-v>X</span>), <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>))</td>
      </tr>
      <tr>
        <td id="L341" class="blob-num js-line-number" data-line-number="341"></td>
        <td id="LC341" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>titles</span> <span class=pl-c1>=</span> [<span class=pl-s1>true</span> <span class=pl-c1>+</span><span class=pl-s>&#39;<span class=pl-cce>\n</span>&#39;</span> <span class=pl-c1>+</span> <span class=pl-s1>pred</span> <span class=pl-k>for</span> <span class=pl-s1>true</span>, <span class=pl-s1>pred</span> <span class=pl-c1>in</span> <span class=pl-en>zip</span>(<span class=pl-s1>trues</span>, <span class=pl-s1>preds</span>)]</td>
      </tr>
      <tr>
        <td id="L342" class="blob-num js-line-number" data-line-number="342"></td>
        <td id="LC342" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>d2l</span>.<span class=pl-en>show_images</span>(<span class=pl-s1>d2l</span>.<span class=pl-en>reshape</span>(<span class=pl-v>X</span>[<span class=pl-c1>0</span>:<span class=pl-s1>n</span>], (<span class=pl-s1>n</span>, <span class=pl-c1>28</span>, <span class=pl-c1>28</span>)), <span class=pl-c1>1</span>, <span class=pl-s1>n</span>, <span class=pl-s1>titles</span><span class=pl-c1>=</span><span class=pl-s1>titles</span>[<span class=pl-c1>0</span>:<span class=pl-s1>n</span>])</td>
      </tr>
      <tr>
        <td id="L343" class="blob-num js-line-number" data-line-number="343"></td>
        <td id="LC343" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L344" class="blob-num js-line-number" data-line-number="344"></td>
        <td id="LC344" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L345" class="blob-num js-line-number" data-line-number="345"></td>
        <td id="LC345" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_multilayer-perceptrons/underfit-overfit.md</span></td>
      </tr>
      <tr>
        <td id="L346" class="blob-num js-line-number" data-line-number="346"></td>
        <td id="LC346" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>evaluate_loss</span>(<span class=pl-s1>net</span>, <span class=pl-s1>data_iter</span>, <span class=pl-s1>loss</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L347" class="blob-num js-line-number" data-line-number="347"></td>
        <td id="LC347" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Evaluate the loss of a model on the given dataset.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L348" class="blob-num js-line-number" data-line-number="348"></td>
        <td id="LC348" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>metric</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Accumulator</span>(<span class=pl-c1>2</span>)  <span class=pl-c># Sum of losses, no. of examples</span></td>
      </tr>
      <tr>
        <td id="L349" class="blob-num js-line-number" data-line-number="349"></td>
        <td id="LC349" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-v>X</span>, <span class=pl-s1>y</span> <span class=pl-c1>in</span> <span class=pl-s1>data_iter</span>:</td>
      </tr>
      <tr>
        <td id="L350" class="blob-num js-line-number" data-line-number="350"></td>
        <td id="LC350" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>out</span> <span class=pl-c1>=</span> <span class=pl-en>net</span>(<span class=pl-v>X</span>)</td>
      </tr>
      <tr>
        <td id="L351" class="blob-num js-line-number" data-line-number="351"></td>
        <td id="LC351" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>reshape</span>(<span class=pl-s1>y</span>, <span class=pl-s1>out</span>.<span class=pl-s1>shape</span>)</td>
      </tr>
      <tr>
        <td id="L352" class="blob-num js-line-number" data-line-number="352"></td>
        <td id="LC352" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>l</span> <span class=pl-c1>=</span> <span class=pl-en>loss</span>(<span class=pl-s1>out</span>, <span class=pl-s1>y</span>)</td>
      </tr>
      <tr>
        <td id="L353" class="blob-num js-line-number" data-line-number="353"></td>
        <td id="LC353" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>metric</span>.<span class=pl-en>add</span>(<span class=pl-s1>d2l</span>.<span class=pl-en>reduce_sum</span>(<span class=pl-s1>l</span>), <span class=pl-s1>d2l</span>.<span class=pl-en>size</span>(<span class=pl-s1>l</span>))</td>
      </tr>
      <tr>
        <td id="L354" class="blob-num js-line-number" data-line-number="354"></td>
        <td id="LC354" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>metric</span>[<span class=pl-c1>0</span>] <span class=pl-c1>/</span> <span class=pl-s1>metric</span>[<span class=pl-c1>1</span>]</td>
      </tr>
      <tr>
        <td id="L355" class="blob-num js-line-number" data-line-number="355"></td>
        <td id="LC355" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L356" class="blob-num js-line-number" data-line-number="356"></td>
        <td id="LC356" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L357" class="blob-num js-line-number" data-line-number="357"></td>
        <td id="LC357" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md</span></td>
      </tr>
      <tr>
        <td id="L358" class="blob-num js-line-number" data-line-number="358"></td>
        <td id="LC358" class="blob-code blob-code-inner js-file-line"><span class=pl-v>DATA_HUB</span> <span class=pl-c1>=</span> <span class=pl-en>dict</span>()  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L359" class="blob-num js-line-number" data-line-number="359"></td>
        <td id="LC359" class="blob-code blob-code-inner js-file-line"><span class=pl-v>DATA_URL</span> <span class=pl-c1>=</span> <span class=pl-s>&#39;http://d2l-data.s3-accelerate.amazonaws.com/&#39;</span>  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L360" class="blob-num js-line-number" data-line-number="360"></td>
        <td id="LC360" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L361" class="blob-num js-line-number" data-line-number="361"></td>
        <td id="LC361" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L362" class="blob-num js-line-number" data-line-number="362"></td>
        <td id="LC362" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md</span></td>
      </tr>
      <tr>
        <td id="L363" class="blob-num js-line-number" data-line-number="363"></td>
        <td id="LC363" class="blob-code blob-code-inner js-file-line"><span class=pl-v>DATA_URL</span> <span class=pl-c1>=</span> <span class=pl-s>&#39;http://d2l-data.s3-accelerate.amazonaws.com/&#39;</span>  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L364" class="blob-num js-line-number" data-line-number="364"></td>
        <td id="LC364" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L365" class="blob-num js-line-number" data-line-number="365"></td>
        <td id="LC365" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L366" class="blob-num js-line-number" data-line-number="366"></td>
        <td id="LC366" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md</span></td>
      </tr>
      <tr>
        <td id="L367" class="blob-num js-line-number" data-line-number="367"></td>
        <td id="LC367" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>download</span>(<span class=pl-s1>name</span>, <span class=pl-s1>cache_dir</span><span class=pl-c1>=</span><span class=pl-s1>os</span>.<span class=pl-s1>path</span>.<span class=pl-en>join</span>(<span class=pl-s>&#39;..&#39;</span>, <span class=pl-s>&#39;data&#39;</span>)):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L368" class="blob-num js-line-number" data-line-number="368"></td>
        <td id="LC368" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Download a file inserted into DATA_HUB, return the local filename.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L369" class="blob-num js-line-number" data-line-number="369"></td>
        <td id="LC369" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>assert</span> <span class=pl-s1>name</span> <span class=pl-c1>in</span> <span class=pl-v>DATA_HUB</span>, <span class=pl-s>f&quot;<span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>name</span><span class=pl-kos>}</span></span> does not exist in <span class=pl-s1><span class=pl-kos>{</span><span class=pl-v>DATA_HUB</span><span class=pl-kos>}</span></span>.&quot;</span></td>
      </tr>
      <tr>
        <td id="L370" class="blob-num js-line-number" data-line-number="370"></td>
        <td id="LC370" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>url</span>, <span class=pl-s1>sha1_hash</span> <span class=pl-c1>=</span> <span class=pl-v>DATA_HUB</span>[<span class=pl-s1>name</span>]</td>
      </tr>
      <tr>
        <td id="L371" class="blob-num js-line-number" data-line-number="371"></td>
        <td id="LC371" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>d2l</span>.<span class=pl-en>mkdir_if_not_exist</span>(<span class=pl-s1>cache_dir</span>)</td>
      </tr>
      <tr>
        <td id="L372" class="blob-num js-line-number" data-line-number="372"></td>
        <td id="LC372" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>fname</span> <span class=pl-c1>=</span> <span class=pl-s1>os</span>.<span class=pl-s1>path</span>.<span class=pl-en>join</span>(<span class=pl-s1>cache_dir</span>, <span class=pl-s1>url</span>.<span class=pl-en>split</span>(<span class=pl-s>&#39;/&#39;</span>)[<span class=pl-c1>-</span><span class=pl-c1>1</span>])</td>
      </tr>
      <tr>
        <td id="L373" class="blob-num js-line-number" data-line-number="373"></td>
        <td id="LC373" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-s1>os</span>.<span class=pl-s1>path</span>.<span class=pl-en>exists</span>(<span class=pl-s1>fname</span>):</td>
      </tr>
      <tr>
        <td id="L374" class="blob-num js-line-number" data-line-number="374"></td>
        <td id="LC374" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>sha1</span> <span class=pl-c1>=</span> <span class=pl-s1>hashlib</span>.<span class=pl-en>sha1</span>()</td>
      </tr>
      <tr>
        <td id="L375" class="blob-num js-line-number" data-line-number="375"></td>
        <td id="LC375" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>with</span> <span class=pl-en>open</span>(<span class=pl-s1>fname</span>, <span class=pl-s>&#39;rb&#39;</span>) <span class=pl-k>as</span> <span class=pl-s1>f</span>:</td>
      </tr>
      <tr>
        <td id="L376" class="blob-num js-line-number" data-line-number="376"></td>
        <td id="LC376" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>while</span> <span class=pl-c1>True</span>:</td>
      </tr>
      <tr>
        <td id="L377" class="blob-num js-line-number" data-line-number="377"></td>
        <td id="LC377" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>data</span> <span class=pl-c1>=</span> <span class=pl-s1>f</span>.<span class=pl-en>read</span>(<span class=pl-c1>1048576</span>)</td>
      </tr>
      <tr>
        <td id="L378" class="blob-num js-line-number" data-line-number="378"></td>
        <td id="LC378" class="blob-code blob-code-inner js-file-line">                <span class=pl-k>if</span> <span class=pl-c1>not</span> <span class=pl-s1>data</span>:</td>
      </tr>
      <tr>
        <td id="L379" class="blob-num js-line-number" data-line-number="379"></td>
        <td id="LC379" class="blob-code blob-code-inner js-file-line">                    <span class=pl-k>break</span></td>
      </tr>
      <tr>
        <td id="L380" class="blob-num js-line-number" data-line-number="380"></td>
        <td id="LC380" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>sha1</span>.<span class=pl-en>update</span>(<span class=pl-s1>data</span>)</td>
      </tr>
      <tr>
        <td id="L381" class="blob-num js-line-number" data-line-number="381"></td>
        <td id="LC381" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>sha1</span>.<span class=pl-en>hexdigest</span>() <span class=pl-c1>==</span> <span class=pl-s1>sha1_hash</span>:</td>
      </tr>
      <tr>
        <td id="L382" class="blob-num js-line-number" data-line-number="382"></td>
        <td id="LC382" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>return</span> <span class=pl-s1>fname</span>  <span class=pl-c># Hit cache</span></td>
      </tr>
      <tr>
        <td id="L383" class="blob-num js-line-number" data-line-number="383"></td>
        <td id="LC383" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>f&#39;Downloading <span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>fname</span><span class=pl-kos>}</span></span> from <span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>url</span><span class=pl-kos>}</span></span>...&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L384" class="blob-num js-line-number" data-line-number="384"></td>
        <td id="LC384" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>r</span> <span class=pl-c1>=</span> <span class=pl-s1>requests</span>.<span class=pl-en>get</span>(<span class=pl-s1>url</span>, <span class=pl-s1>stream</span><span class=pl-c1>=</span><span class=pl-c1>True</span>, <span class=pl-s1>verify</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L385" class="blob-num js-line-number" data-line-number="385"></td>
        <td id="LC385" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>with</span> <span class=pl-en>open</span>(<span class=pl-s1>fname</span>, <span class=pl-s>&#39;wb&#39;</span>) <span class=pl-k>as</span> <span class=pl-s1>f</span>:</td>
      </tr>
      <tr>
        <td id="L386" class="blob-num js-line-number" data-line-number="386"></td>
        <td id="LC386" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>f</span>.<span class=pl-en>write</span>(<span class=pl-s1>r</span>.<span class=pl-s1>content</span>)</td>
      </tr>
      <tr>
        <td id="L387" class="blob-num js-line-number" data-line-number="387"></td>
        <td id="LC387" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>fname</span></td>
      </tr>
      <tr>
        <td id="L388" class="blob-num js-line-number" data-line-number="388"></td>
        <td id="LC388" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L389" class="blob-num js-line-number" data-line-number="389"></td>
        <td id="LC389" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L390" class="blob-num js-line-number" data-line-number="390"></td>
        <td id="LC390" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md</span></td>
      </tr>
      <tr>
        <td id="L391" class="blob-num js-line-number" data-line-number="391"></td>
        <td id="LC391" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>download_extract</span>(<span class=pl-s1>name</span>, <span class=pl-s1>folder</span><span class=pl-c1>=</span><span class=pl-c1>None</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L392" class="blob-num js-line-number" data-line-number="392"></td>
        <td id="LC392" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Download and extract a zip/tar file.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L393" class="blob-num js-line-number" data-line-number="393"></td>
        <td id="LC393" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>fname</span> <span class=pl-c1>=</span> <span class=pl-en>download</span>(<span class=pl-s1>name</span>)</td>
      </tr>
      <tr>
        <td id="L394" class="blob-num js-line-number" data-line-number="394"></td>
        <td id="LC394" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>base_dir</span> <span class=pl-c1>=</span> <span class=pl-s1>os</span>.<span class=pl-s1>path</span>.<span class=pl-en>dirname</span>(<span class=pl-s1>fname</span>)</td>
      </tr>
      <tr>
        <td id="L395" class="blob-num js-line-number" data-line-number="395"></td>
        <td id="LC395" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>data_dir</span>, <span class=pl-s1>ext</span> <span class=pl-c1>=</span> <span class=pl-s1>os</span>.<span class=pl-s1>path</span>.<span class=pl-en>splitext</span>(<span class=pl-s1>fname</span>)</td>
      </tr>
      <tr>
        <td id="L396" class="blob-num js-line-number" data-line-number="396"></td>
        <td id="LC396" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-s1>ext</span> <span class=pl-c1>==</span> <span class=pl-s>&#39;.zip&#39;</span>:</td>
      </tr>
      <tr>
        <td id="L397" class="blob-num js-line-number" data-line-number="397"></td>
        <td id="LC397" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>fp</span> <span class=pl-c1>=</span> <span class=pl-s1>zipfile</span>.<span class=pl-v>ZipFile</span>(<span class=pl-s1>fname</span>, <span class=pl-s>&#39;r&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L398" class="blob-num js-line-number" data-line-number="398"></td>
        <td id="LC398" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>elif</span> <span class=pl-s1>ext</span> <span class=pl-c1>in</span> (<span class=pl-s>&#39;.tar&#39;</span>, <span class=pl-s>&#39;.gz&#39;</span>):</td>
      </tr>
      <tr>
        <td id="L399" class="blob-num js-line-number" data-line-number="399"></td>
        <td id="LC399" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>fp</span> <span class=pl-c1>=</span> <span class=pl-s1>tarfile</span>.<span class=pl-en>open</span>(<span class=pl-s1>fname</span>, <span class=pl-s>&#39;r&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L400" class="blob-num js-line-number" data-line-number="400"></td>
        <td id="LC400" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>else</span>:</td>
      </tr>
      <tr>
        <td id="L401" class="blob-num js-line-number" data-line-number="401"></td>
        <td id="LC401" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>assert</span> <span class=pl-c1>False</span>, <span class=pl-s>&#39;Only zip/tar files can be extracted.&#39;</span></td>
      </tr>
      <tr>
        <td id="L402" class="blob-num js-line-number" data-line-number="402"></td>
        <td id="LC402" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>fp</span>.<span class=pl-en>extractall</span>(<span class=pl-s1>base_dir</span>)</td>
      </tr>
      <tr>
        <td id="L403" class="blob-num js-line-number" data-line-number="403"></td>
        <td id="LC403" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>os</span>.<span class=pl-s1>path</span>.<span class=pl-en>join</span>(<span class=pl-s1>base_dir</span>, <span class=pl-s1>folder</span>) <span class=pl-k>if</span> <span class=pl-s1>folder</span> <span class=pl-k>else</span> <span class=pl-s1>data_dir</span></td>
      </tr>
      <tr>
        <td id="L404" class="blob-num js-line-number" data-line-number="404"></td>
        <td id="LC404" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L405" class="blob-num js-line-number" data-line-number="405"></td>
        <td id="LC405" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L406" class="blob-num js-line-number" data-line-number="406"></td>
        <td id="LC406" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md</span></td>
      </tr>
      <tr>
        <td id="L407" class="blob-num js-line-number" data-line-number="407"></td>
        <td id="LC407" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>download_all</span>():  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L408" class="blob-num js-line-number" data-line-number="408"></td>
        <td id="LC408" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Download all files in the DATA_HUB.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L409" class="blob-num js-line-number" data-line-number="409"></td>
        <td id="LC409" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>name</span> <span class=pl-c1>in</span> <span class=pl-v>DATA_HUB</span>:</td>
      </tr>
      <tr>
        <td id="L410" class="blob-num js-line-number" data-line-number="410"></td>
        <td id="LC410" class="blob-code blob-code-inner js-file-line">        <span class=pl-en>download</span>(<span class=pl-s1>name</span>)</td>
      </tr>
      <tr>
        <td id="L411" class="blob-num js-line-number" data-line-number="411"></td>
        <td id="LC411" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L412" class="blob-num js-line-number" data-line-number="412"></td>
        <td id="LC412" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L413" class="blob-num js-line-number" data-line-number="413"></td>
        <td id="LC413" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md</span></td>
      </tr>
      <tr>
        <td id="L414" class="blob-num js-line-number" data-line-number="414"></td>
        <td id="LC414" class="blob-code blob-code-inner js-file-line"><span class=pl-v>DATA_HUB</span>[<span class=pl-s>&#39;kaggle_house_train&#39;</span>] <span class=pl-c1>=</span> (  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L415" class="blob-num js-line-number" data-line-number="415"></td>
        <td id="LC415" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>DATA_URL</span> <span class=pl-c1>+</span> <span class=pl-s>&#39;kaggle_house_pred_train.csv&#39;</span>,</td>
      </tr>
      <tr>
        <td id="L416" class="blob-num js-line-number" data-line-number="416"></td>
        <td id="LC416" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&#39;585e9cc93e70b39160e7921475f9bcd7d31219ce&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L417" class="blob-num js-line-number" data-line-number="417"></td>
        <td id="LC417" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L418" class="blob-num js-line-number" data-line-number="418"></td>
        <td id="LC418" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L419" class="blob-num js-line-number" data-line-number="419"></td>
        <td id="LC419" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md</span></td>
      </tr>
      <tr>
        <td id="L420" class="blob-num js-line-number" data-line-number="420"></td>
        <td id="LC420" class="blob-code blob-code-inner js-file-line"><span class=pl-v>DATA_HUB</span>[<span class=pl-s>&#39;kaggle_house_test&#39;</span>] <span class=pl-c1>=</span> (  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L421" class="blob-num js-line-number" data-line-number="421"></td>
        <td id="LC421" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>DATA_URL</span> <span class=pl-c1>+</span> <span class=pl-s>&#39;kaggle_house_pred_test.csv&#39;</span>,</td>
      </tr>
      <tr>
        <td id="L422" class="blob-num js-line-number" data-line-number="422"></td>
        <td id="LC422" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&#39;fa19780a7b011d9b009e8bff8e99922a8ee2eb90&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L423" class="blob-num js-line-number" data-line-number="423"></td>
        <td id="LC423" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L424" class="blob-num js-line-number" data-line-number="424"></td>
        <td id="LC424" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L425" class="blob-num js-line-number" data-line-number="425"></td>
        <td id="LC425" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_deep-learning-computation/use-gpu.md</span></td>
      </tr>
      <tr>
        <td id="L426" class="blob-num js-line-number" data-line-number="426"></td>
        <td id="LC426" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>try_gpu</span>(<span class=pl-s1>i</span><span class=pl-c1>=</span><span class=pl-c1>0</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L427" class="blob-num js-line-number" data-line-number="427"></td>
        <td id="LC427" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Return gpu(i) if exists, otherwise return cpu().&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L428" class="blob-num js-line-number" data-line-number="428"></td>
        <td id="LC428" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-s1>torch</span>.<span class=pl-s1>cuda</span>.<span class=pl-en>device_count</span>() <span class=pl-c1>&gt;=</span> <span class=pl-s1>i</span> <span class=pl-c1>+</span> <span class=pl-c1>1</span>:</td>
      </tr>
      <tr>
        <td id="L429" class="blob-num js-line-number" data-line-number="429"></td>
        <td id="LC429" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>torch</span>.<span class=pl-en>device</span>(<span class=pl-s>f&#39;cuda:<span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>i</span><span class=pl-kos>}</span></span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L430" class="blob-num js-line-number" data-line-number="430"></td>
        <td id="LC430" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>torch</span>.<span class=pl-en>device</span>(<span class=pl-s>&#39;cpu&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L431" class="blob-num js-line-number" data-line-number="431"></td>
        <td id="LC431" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L432" class="blob-num js-line-number" data-line-number="432"></td>
        <td id="LC432" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L433" class="blob-num js-line-number" data-line-number="433"></td>
        <td id="LC433" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_deep-learning-computation/use-gpu.md</span></td>
      </tr>
      <tr>
        <td id="L434" class="blob-num js-line-number" data-line-number="434"></td>
        <td id="LC434" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>try_all_gpus</span>():  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L435" class="blob-num js-line-number" data-line-number="435"></td>
        <td id="LC435" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Return all available GPUs, or [cpu(),] if no GPU exists.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L436" class="blob-num js-line-number" data-line-number="436"></td>
        <td id="LC436" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>devices</span> <span class=pl-c1>=</span> [<span class=pl-s1>torch</span>.<span class=pl-en>device</span>(<span class=pl-s>f&#39;cuda:<span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>i</span><span class=pl-kos>}</span></span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L437" class="blob-num js-line-number" data-line-number="437"></td>
        <td id="LC437" class="blob-code blob-code-inner js-file-line">             <span class=pl-k>for</span> <span class=pl-s1>i</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-s1>torch</span>.<span class=pl-s1>cuda</span>.<span class=pl-en>device_count</span>())]</td>
      </tr>
      <tr>
        <td id="L438" class="blob-num js-line-number" data-line-number="438"></td>
        <td id="LC438" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>devices</span> <span class=pl-k>if</span> <span class=pl-s1>devices</span> <span class=pl-k>else</span> [<span class=pl-s1>torch</span>.<span class=pl-en>device</span>(<span class=pl-s>&#39;cpu&#39;</span>)]</td>
      </tr>
      <tr>
        <td id="L439" class="blob-num js-line-number" data-line-number="439"></td>
        <td id="LC439" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L440" class="blob-num js-line-number" data-line-number="440"></td>
        <td id="LC440" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L441" class="blob-num js-line-number" data-line-number="441"></td>
        <td id="LC441" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_convolutional-neural-networks/conv-layer.md</span></td>
      </tr>
      <tr>
        <td id="L442" class="blob-num js-line-number" data-line-number="442"></td>
        <td id="LC442" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>corr2d</span>(<span class=pl-v>X</span>, <span class=pl-v>K</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L443" class="blob-num js-line-number" data-line-number="443"></td>
        <td id="LC443" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Compute 2D cross-correlation.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L444" class="blob-num js-line-number" data-line-number="444"></td>
        <td id="LC444" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>h</span>, <span class=pl-s1>w</span> <span class=pl-c1>=</span> <span class=pl-v>K</span>.<span class=pl-s1>shape</span></td>
      </tr>
      <tr>
        <td id="L445" class="blob-num js-line-number" data-line-number="445"></td>
        <td id="LC445" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Y</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>zeros</span>((<span class=pl-v>X</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>0</span>] <span class=pl-c1>-</span> <span class=pl-s1>h</span> <span class=pl-c1>+</span> <span class=pl-c1>1</span>, <span class=pl-v>X</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>1</span>] <span class=pl-c1>-</span> <span class=pl-s1>w</span> <span class=pl-c1>+</span> <span class=pl-c1>1</span>))</td>
      </tr>
      <tr>
        <td id="L446" class="blob-num js-line-number" data-line-number="446"></td>
        <td id="LC446" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>i</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-v>Y</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>0</span>]):</td>
      </tr>
      <tr>
        <td id="L447" class="blob-num js-line-number" data-line-number="447"></td>
        <td id="LC447" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-s1>j</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-v>Y</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>1</span>]):</td>
      </tr>
      <tr>
        <td id="L448" class="blob-num js-line-number" data-line-number="448"></td>
        <td id="LC448" class="blob-code blob-code-inner js-file-line">            <span class=pl-v>Y</span>[<span class=pl-s1>i</span>, <span class=pl-s1>j</span>] <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>reduce_sum</span>((<span class=pl-v>X</span>[<span class=pl-s1>i</span>: <span class=pl-s1>i</span> <span class=pl-c1>+</span> <span class=pl-s1>h</span>, <span class=pl-s1>j</span>: <span class=pl-s1>j</span> <span class=pl-c1>+</span> <span class=pl-s1>w</span>] <span class=pl-c1>*</span> <span class=pl-v>K</span>))</td>
      </tr>
      <tr>
        <td id="L449" class="blob-num js-line-number" data-line-number="449"></td>
        <td id="LC449" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-v>Y</span></td>
      </tr>
      <tr>
        <td id="L450" class="blob-num js-line-number" data-line-number="450"></td>
        <td id="LC450" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L451" class="blob-num js-line-number" data-line-number="451"></td>
        <td id="LC451" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L452" class="blob-num js-line-number" data-line-number="452"></td>
        <td id="LC452" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_convolutional-neural-networks/lenet.md</span></td>
      </tr>
      <tr>
        <td id="L453" class="blob-num js-line-number" data-line-number="453"></td>
        <td id="LC453" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>evaluate_accuracy_gpu</span>(<span class=pl-s1>net</span>, <span class=pl-s1>data_iter</span>, <span class=pl-s1>device</span><span class=pl-c1>=</span><span class=pl-c1>None</span>): <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L454" class="blob-num js-line-number" data-line-number="454"></td>
        <td id="LC454" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Compute the accuracy for a model on a dataset using a GPU.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L455" class="blob-num js-line-number" data-line-number="455"></td>
        <td id="LC455" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>net</span>.<span class=pl-en>eval</span>()  <span class=pl-c># Set the model to evaluation mode</span></td>
      </tr>
      <tr>
        <td id="L456" class="blob-num js-line-number" data-line-number="456"></td>
        <td id="LC456" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-c1>not</span> <span class=pl-s1>device</span>:</td>
      </tr>
      <tr>
        <td id="L457" class="blob-num js-line-number" data-line-number="457"></td>
        <td id="LC457" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>device</span> <span class=pl-c1>=</span> <span class=pl-en>next</span>(<span class=pl-en>iter</span>(<span class=pl-s1>net</span>.<span class=pl-en>parameters</span>())).<span class=pl-s1>device</span></td>
      </tr>
      <tr>
        <td id="L458" class="blob-num js-line-number" data-line-number="458"></td>
        <td id="LC458" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># No. of correct predictions, no. of predictions</span></td>
      </tr>
      <tr>
        <td id="L459" class="blob-num js-line-number" data-line-number="459"></td>
        <td id="LC459" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>metric</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Accumulator</span>(<span class=pl-c1>2</span>)</td>
      </tr>
      <tr>
        <td id="L460" class="blob-num js-line-number" data-line-number="460"></td>
        <td id="LC460" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-v>X</span>, <span class=pl-s1>y</span> <span class=pl-c1>in</span> <span class=pl-s1>data_iter</span>:</td>
      </tr>
      <tr>
        <td id="L461" class="blob-num js-line-number" data-line-number="461"></td>
        <td id="LC461" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>X</span>, <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-v>X</span>.<span class=pl-en>to</span>(<span class=pl-s1>device</span>), <span class=pl-s1>y</span>.<span class=pl-en>to</span>(<span class=pl-s1>device</span>)</td>
      </tr>
      <tr>
        <td id="L462" class="blob-num js-line-number" data-line-number="462"></td>
        <td id="LC462" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>metric</span>.<span class=pl-en>add</span>(<span class=pl-s1>d2l</span>.<span class=pl-en>accuracy</span>(<span class=pl-en>net</span>(<span class=pl-v>X</span>), <span class=pl-s1>y</span>), <span class=pl-s1>d2l</span>.<span class=pl-en>size</span>(<span class=pl-s1>y</span>))</td>
      </tr>
      <tr>
        <td id="L463" class="blob-num js-line-number" data-line-number="463"></td>
        <td id="LC463" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>metric</span>[<span class=pl-c1>0</span>] <span class=pl-c1>/</span> <span class=pl-s1>metric</span>[<span class=pl-c1>1</span>]</td>
      </tr>
      <tr>
        <td id="L464" class="blob-num js-line-number" data-line-number="464"></td>
        <td id="LC464" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L465" class="blob-num js-line-number" data-line-number="465"></td>
        <td id="LC465" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L466" class="blob-num js-line-number" data-line-number="466"></td>
        <td id="LC466" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_convolutional-neural-networks/lenet.md</span></td>
      </tr>
      <tr>
        <td id="L467" class="blob-num js-line-number" data-line-number="467"></td>
        <td id="LC467" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>train_ch6</span>(<span class=pl-s1>net</span>, <span class=pl-s1>train_iter</span>, <span class=pl-s1>test_iter</span>, <span class=pl-s1>num_epochs</span>, <span class=pl-s1>lr</span>,</td>
      </tr>
      <tr>
        <td id="L468" class="blob-num js-line-number" data-line-number="468"></td>
        <td id="LC468" class="blob-code blob-code-inner js-file-line">              <span class=pl-s1>device</span><span class=pl-c1>=</span><span class=pl-s1>d2l</span>.<span class=pl-en>try_gpu</span>()):</td>
      </tr>
      <tr>
        <td id="L469" class="blob-num js-line-number" data-line-number="469"></td>
        <td id="LC469" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Train a model with a GPU (defined in Chapter 6).&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L470" class="blob-num js-line-number" data-line-number="470"></td>
        <td id="LC470" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>init_weights</span>(<span class=pl-s1>m</span>):</td>
      </tr>
      <tr>
        <td id="L471" class="blob-num js-line-number" data-line-number="471"></td>
        <td id="LC471" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-en>type</span>(<span class=pl-s1>m</span>) <span class=pl-c1>==</span> <span class=pl-s1>nn</span>.<span class=pl-v>Linear</span> <span class=pl-c1>or</span> <span class=pl-en>type</span>(<span class=pl-s1>m</span>) <span class=pl-c1>==</span> <span class=pl-s1>nn</span>.<span class=pl-v>Conv2d</span>:</td>
      </tr>
      <tr>
        <td id="L472" class="blob-num js-line-number" data-line-number="472"></td>
        <td id="LC472" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>torch</span>.<span class=pl-s1>nn</span>.<span class=pl-s1>init</span>.<span class=pl-en>xavier_uniform_</span>(<span class=pl-s1>m</span>.<span class=pl-s1>weight</span>)</td>
      </tr>
      <tr>
        <td id="L473" class="blob-num js-line-number" data-line-number="473"></td>
        <td id="LC473" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>net</span>.<span class=pl-en>apply</span>(<span class=pl-s1>init_weights</span>)</td>
      </tr>
      <tr>
        <td id="L474" class="blob-num js-line-number" data-line-number="474"></td>
        <td id="LC474" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;training on&#39;</span>, <span class=pl-s1>device</span>)</td>
      </tr>
      <tr>
        <td id="L475" class="blob-num js-line-number" data-line-number="475"></td>
        <td id="LC475" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>net</span>.<span class=pl-en>to</span>(<span class=pl-s1>device</span>)</td>
      </tr>
      <tr>
        <td id="L476" class="blob-num js-line-number" data-line-number="476"></td>
        <td id="LC476" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>optimizer</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>optim</span>.<span class=pl-v>SGD</span>(<span class=pl-s1>net</span>.<span class=pl-en>parameters</span>(), <span class=pl-s1>lr</span><span class=pl-c1>=</span><span class=pl-s1>lr</span>)</td>
      </tr>
      <tr>
        <td id="L477" class="blob-num js-line-number" data-line-number="477"></td>
        <td id="LC477" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>loss</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>CrossEntropyLoss</span>()</td>
      </tr>
      <tr>
        <td id="L478" class="blob-num js-line-number" data-line-number="478"></td>
        <td id="LC478" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>animator</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Animator</span>(<span class=pl-s1>xlabel</span><span class=pl-c1>=</span><span class=pl-s>&#39;epoch&#39;</span>, <span class=pl-s1>xlim</span><span class=pl-c1>=</span>[<span class=pl-c1>0</span>, <span class=pl-s1>num_epochs</span>],</td>
      </tr>
      <tr>
        <td id="L479" class="blob-num js-line-number" data-line-number="479"></td>
        <td id="LC479" class="blob-code blob-code-inner js-file-line">                            <span class=pl-s1>legend</span><span class=pl-c1>=</span>[<span class=pl-s>&#39;train loss&#39;</span>, <span class=pl-s>&#39;train acc&#39;</span>, <span class=pl-s>&#39;test acc&#39;</span>])</td>
      </tr>
      <tr>
        <td id="L480" class="blob-num js-line-number" data-line-number="480"></td>
        <td id="LC480" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>timer</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Timer</span>()</td>
      </tr>
      <tr>
        <td id="L481" class="blob-num js-line-number" data-line-number="481"></td>
        <td id="LC481" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>epoch</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-s1>num_epochs</span>):</td>
      </tr>
      <tr>
        <td id="L482" class="blob-num js-line-number" data-line-number="482"></td>
        <td id="LC482" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># Sum of training loss, sum of training accuracy, no. of examples</span></td>
      </tr>
      <tr>
        <td id="L483" class="blob-num js-line-number" data-line-number="483"></td>
        <td id="LC483" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>metric</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Accumulator</span>(<span class=pl-c1>3</span>)</td>
      </tr>
      <tr>
        <td id="L484" class="blob-num js-line-number" data-line-number="484"></td>
        <td id="LC484" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-s1>i</span>, (<span class=pl-v>X</span>, <span class=pl-s1>y</span>) <span class=pl-c1>in</span> <span class=pl-en>enumerate</span>(<span class=pl-s1>train_iter</span>):</td>
      </tr>
      <tr>
        <td id="L485" class="blob-num js-line-number" data-line-number="485"></td>
        <td id="LC485" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>timer</span>.<span class=pl-en>start</span>()</td>
      </tr>
      <tr>
        <td id="L486" class="blob-num js-line-number" data-line-number="486"></td>
        <td id="LC486" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>net</span>.<span class=pl-en>train</span>()</td>
      </tr>
      <tr>
        <td id="L487" class="blob-num js-line-number" data-line-number="487"></td>
        <td id="LC487" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>optimizer</span>.<span class=pl-en>zero_grad</span>()</td>
      </tr>
      <tr>
        <td id="L488" class="blob-num js-line-number" data-line-number="488"></td>
        <td id="LC488" class="blob-code blob-code-inner js-file-line">            <span class=pl-v>X</span>, <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-v>X</span>.<span class=pl-en>to</span>(<span class=pl-s1>device</span>), <span class=pl-s1>y</span>.<span class=pl-en>to</span>(<span class=pl-s1>device</span>)</td>
      </tr>
      <tr>
        <td id="L489" class="blob-num js-line-number" data-line-number="489"></td>
        <td id="LC489" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>y_hat</span> <span class=pl-c1>=</span> <span class=pl-en>net</span>(<span class=pl-v>X</span>)</td>
      </tr>
      <tr>
        <td id="L490" class="blob-num js-line-number" data-line-number="490"></td>
        <td id="LC490" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>l</span> <span class=pl-c1>=</span> <span class=pl-en>loss</span>(<span class=pl-s1>y_hat</span>, <span class=pl-s1>y</span>)</td>
      </tr>
      <tr>
        <td id="L491" class="blob-num js-line-number" data-line-number="491"></td>
        <td id="LC491" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>l</span>.<span class=pl-en>backward</span>()</td>
      </tr>
      <tr>
        <td id="L492" class="blob-num js-line-number" data-line-number="492"></td>
        <td id="LC492" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>optimizer</span>.<span class=pl-en>step</span>()</td>
      </tr>
      <tr>
        <td id="L493" class="blob-num js-line-number" data-line-number="493"></td>
        <td id="LC493" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>with</span> <span class=pl-s1>torch</span>.<span class=pl-en>no_grad</span>():</td>
      </tr>
      <tr>
        <td id="L494" class="blob-num js-line-number" data-line-number="494"></td>
        <td id="LC494" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>metric</span>.<span class=pl-en>add</span>(<span class=pl-s1>l</span> <span class=pl-c1>*</span> <span class=pl-v>X</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>0</span>], <span class=pl-s1>d2l</span>.<span class=pl-en>accuracy</span>(<span class=pl-s1>y_hat</span>, <span class=pl-s1>y</span>), <span class=pl-v>X</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>0</span>])</td>
      </tr>
      <tr>
        <td id="L495" class="blob-num js-line-number" data-line-number="495"></td>
        <td id="LC495" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>timer</span>.<span class=pl-en>stop</span>()</td>
      </tr>
      <tr>
        <td id="L496" class="blob-num js-line-number" data-line-number="496"></td>
        <td id="LC496" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>train_loss</span> <span class=pl-c1>=</span> <span class=pl-s1>metric</span>[<span class=pl-c1>0</span>]<span class=pl-c1>/</span><span class=pl-s1>metric</span>[<span class=pl-c1>2</span>]</td>
      </tr>
      <tr>
        <td id="L497" class="blob-num js-line-number" data-line-number="497"></td>
        <td id="LC497" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>train_acc</span> <span class=pl-c1>=</span> <span class=pl-s1>metric</span>[<span class=pl-c1>1</span>]<span class=pl-c1>/</span><span class=pl-s1>metric</span>[<span class=pl-c1>2</span>]</td>
      </tr>
      <tr>
        <td id="L498" class="blob-num js-line-number" data-line-number="498"></td>
        <td id="LC498" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>if</span> (<span class=pl-s1>i</span> <span class=pl-c1>+</span> <span class=pl-c1>1</span>) <span class=pl-c1>%</span> <span class=pl-c1>50</span> <span class=pl-c1>==</span> <span class=pl-c1>0</span>:</td>
      </tr>
      <tr>
        <td id="L499" class="blob-num js-line-number" data-line-number="499"></td>
        <td id="LC499" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>animator</span>.<span class=pl-en>add</span>(<span class=pl-s1>epoch</span> <span class=pl-c1>+</span> <span class=pl-s1>i</span> <span class=pl-c1>/</span> <span class=pl-en>len</span>(<span class=pl-s1>train_iter</span>),</td>
      </tr>
      <tr>
        <td id="L500" class="blob-num js-line-number" data-line-number="500"></td>
        <td id="LC500" class="blob-code blob-code-inner js-file-line">                             (<span class=pl-s1>train_loss</span>, <span class=pl-s1>train_acc</span>, <span class=pl-c1>None</span>))</td>
      </tr>
      <tr>
        <td id="L501" class="blob-num js-line-number" data-line-number="501"></td>
        <td id="LC501" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>test_acc</span> <span class=pl-c1>=</span> <span class=pl-en>evaluate_accuracy_gpu</span>(<span class=pl-s1>net</span>, <span class=pl-s1>test_iter</span>)</td>
      </tr>
      <tr>
        <td id="L502" class="blob-num js-line-number" data-line-number="502"></td>
        <td id="LC502" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>animator</span>.<span class=pl-en>add</span>(<span class=pl-s1>epoch</span><span class=pl-c1>+</span><span class=pl-c1>1</span>, (<span class=pl-c1>None</span>, <span class=pl-c1>None</span>, <span class=pl-s1>test_acc</span>))</td>
      </tr>
      <tr>
        <td id="L503" class="blob-num js-line-number" data-line-number="503"></td>
        <td id="LC503" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>f&#39;loss <span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>train_loss</span>:.3f<span class=pl-kos>}</span></span>, train acc <span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>train_acc</span>:.3f<span class=pl-kos>}</span></span>, &#39;</span></td>
      </tr>
      <tr>
        <td id="L504" class="blob-num js-line-number" data-line-number="504"></td>
        <td id="LC504" class="blob-code blob-code-inner js-file-line">          <span class=pl-s>f&#39;test acc <span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>test_acc</span>:.3f<span class=pl-kos>}</span></span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L505" class="blob-num js-line-number" data-line-number="505"></td>
        <td id="LC505" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>f&#39;<span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>metric</span>[<span class=pl-c1>2</span>] <span class=pl-c1>*</span> <span class=pl-s1>num_epochs</span> <span class=pl-c1>/</span> <span class=pl-s1>timer</span>.<span class=pl-en>sum</span>():.1f<span class=pl-kos>}</span></span> examples/sec &#39;</span></td>
      </tr>
      <tr>
        <td id="L506" class="blob-num js-line-number" data-line-number="506"></td>
        <td id="LC506" class="blob-code blob-code-inner js-file-line">          <span class=pl-s>f&#39;on <span class=pl-s1><span class=pl-kos>{</span><span class=pl-en>str</span>(<span class=pl-s1>device</span>)<span class=pl-kos>}</span></span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L507" class="blob-num js-line-number" data-line-number="507"></td>
        <td id="LC507" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L508" class="blob-num js-line-number" data-line-number="508"></td>
        <td id="LC508" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L509" class="blob-num js-line-number" data-line-number="509"></td>
        <td id="LC509" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_convolutional-modern/resnet.md</span></td>
      </tr>
      <tr>
        <td id="L510" class="blob-num js-line-number" data-line-number="510"></td>
        <td id="LC510" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>Residual</span>(<span class=pl-s1>nn</span>.<span class=pl-v>Module</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L511" class="blob-num js-line-number" data-line-number="511"></td>
        <td id="LC511" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>, <span class=pl-s1>input_channels</span>, <span class=pl-s1>num_channels</span>,</td>
      </tr>
      <tr>
        <td id="L512" class="blob-num js-line-number" data-line-number="512"></td>
        <td id="LC512" class="blob-code blob-code-inner js-file-line">                 <span class=pl-s1>use_1x1conv</span><span class=pl-c1>=</span><span class=pl-c1>False</span>, <span class=pl-s1>strides</span><span class=pl-c1>=</span><span class=pl-c1>1</span>):</td>
      </tr>
      <tr>
        <td id="L513" class="blob-num js-line-number" data-line-number="513"></td>
        <td id="LC513" class="blob-code blob-code-inner js-file-line">        <span class=pl-en>super</span>().<span class=pl-en>__init__</span>()</td>
      </tr>
      <tr>
        <td id="L514" class="blob-num js-line-number" data-line-number="514"></td>
        <td id="LC514" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>conv1</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>Conv2d</span>(<span class=pl-s1>input_channels</span>, <span class=pl-s1>num_channels</span>,</td>
      </tr>
      <tr>
        <td id="L515" class="blob-num js-line-number" data-line-number="515"></td>
        <td id="LC515" class="blob-code blob-code-inner js-file-line">                               <span class=pl-s1>kernel_size</span><span class=pl-c1>=</span><span class=pl-c1>3</span>, <span class=pl-s1>padding</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>stride</span><span class=pl-c1>=</span><span class=pl-s1>strides</span>)</td>
      </tr>
      <tr>
        <td id="L516" class="blob-num js-line-number" data-line-number="516"></td>
        <td id="LC516" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>conv2</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>Conv2d</span>(<span class=pl-s1>num_channels</span>, <span class=pl-s1>num_channels</span>,</td>
      </tr>
      <tr>
        <td id="L517" class="blob-num js-line-number" data-line-number="517"></td>
        <td id="LC517" class="blob-code blob-code-inner js-file-line">                               <span class=pl-s1>kernel_size</span><span class=pl-c1>=</span><span class=pl-c1>3</span>, <span class=pl-s1>padding</span><span class=pl-c1>=</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L518" class="blob-num js-line-number" data-line-number="518"></td>
        <td id="LC518" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>use_1x1conv</span>:</td>
      </tr>
      <tr>
        <td id="L519" class="blob-num js-line-number" data-line-number="519"></td>
        <td id="LC519" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>self</span>.<span class=pl-s1>conv3</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>Conv2d</span>(<span class=pl-s1>input_channels</span>, <span class=pl-s1>num_channels</span>,</td>
      </tr>
      <tr>
        <td id="L520" class="blob-num js-line-number" data-line-number="520"></td>
        <td id="LC520" class="blob-code blob-code-inner js-file-line">                                   <span class=pl-s1>kernel_size</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>stride</span><span class=pl-c1>=</span><span class=pl-s1>strides</span>)</td>
      </tr>
      <tr>
        <td id="L521" class="blob-num js-line-number" data-line-number="521"></td>
        <td id="LC521" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>else</span>:</td>
      </tr>
      <tr>
        <td id="L522" class="blob-num js-line-number" data-line-number="522"></td>
        <td id="LC522" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>self</span>.<span class=pl-s1>conv3</span> <span class=pl-c1>=</span> <span class=pl-c1>None</span></td>
      </tr>
      <tr>
        <td id="L523" class="blob-num js-line-number" data-line-number="523"></td>
        <td id="LC523" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>bn1</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>BatchNorm2d</span>(<span class=pl-s1>num_channels</span>)</td>
      </tr>
      <tr>
        <td id="L524" class="blob-num js-line-number" data-line-number="524"></td>
        <td id="LC524" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>bn2</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>BatchNorm2d</span>(<span class=pl-s1>num_channels</span>)</td>
      </tr>
      <tr>
        <td id="L525" class="blob-num js-line-number" data-line-number="525"></td>
        <td id="LC525" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>relu</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>ReLU</span>(<span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L526" class="blob-num js-line-number" data-line-number="526"></td>
        <td id="LC526" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L527" class="blob-num js-line-number" data-line-number="527"></td>
        <td id="LC527" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>forward</span>(<span class=pl-s1>self</span>, <span class=pl-v>X</span>):</td>
      </tr>
      <tr>
        <td id="L528" class="blob-num js-line-number" data-line-number="528"></td>
        <td id="LC528" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>Y</span> <span class=pl-c1>=</span> <span class=pl-v>F</span>.<span class=pl-en>relu</span>(<span class=pl-s1>self</span>.<span class=pl-en>bn1</span>(<span class=pl-s1>self</span>.<span class=pl-en>conv1</span>(<span class=pl-v>X</span>)))</td>
      </tr>
      <tr>
        <td id="L529" class="blob-num js-line-number" data-line-number="529"></td>
        <td id="LC529" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>Y</span> <span class=pl-c1>=</span> <span class=pl-s1>self</span>.<span class=pl-en>bn2</span>(<span class=pl-s1>self</span>.<span class=pl-en>conv2</span>(<span class=pl-v>Y</span>))</td>
      </tr>
      <tr>
        <td id="L530" class="blob-num js-line-number" data-line-number="530"></td>
        <td id="LC530" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>self</span>.<span class=pl-s1>conv3</span>:</td>
      </tr>
      <tr>
        <td id="L531" class="blob-num js-line-number" data-line-number="531"></td>
        <td id="LC531" class="blob-code blob-code-inner js-file-line">            <span class=pl-v>X</span> <span class=pl-c1>=</span> <span class=pl-s1>self</span>.<span class=pl-en>conv3</span>(<span class=pl-v>X</span>)</td>
      </tr>
      <tr>
        <td id="L532" class="blob-num js-line-number" data-line-number="532"></td>
        <td id="LC532" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>Y</span> <span class=pl-c1>+=</span> <span class=pl-v>X</span></td>
      </tr>
      <tr>
        <td id="L533" class="blob-num js-line-number" data-line-number="533"></td>
        <td id="LC533" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-v>F</span>.<span class=pl-en>relu</span>(<span class=pl-v>Y</span>)</td>
      </tr>
      <tr>
        <td id="L534" class="blob-num js-line-number" data-line-number="534"></td>
        <td id="LC534" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L535" class="blob-num js-line-number" data-line-number="535"></td>
        <td id="LC535" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L536" class="blob-num js-line-number" data-line-number="536"></td>
        <td id="LC536" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md</span></td>
      </tr>
      <tr>
        <td id="L537" class="blob-num js-line-number" data-line-number="537"></td>
        <td id="LC537" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>d2l</span>.<span class=pl-v>DATA_HUB</span>[<span class=pl-s>&#39;time_machine&#39;</span>] <span class=pl-c1>=</span> (<span class=pl-s1>d2l</span>.<span class=pl-v>DATA_URL</span> <span class=pl-c1>+</span> <span class=pl-s>&#39;timemachine.txt&#39;</span>,</td>
      </tr>
      <tr>
        <td id="L538" class="blob-num js-line-number" data-line-number="538"></td>
        <td id="LC538" class="blob-code blob-code-inner js-file-line">                                <span class=pl-s>&#39;090b5e7e70c295757f55df93cb0a180b9691891a&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L539" class="blob-num js-line-number" data-line-number="539"></td>
        <td id="LC539" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L540" class="blob-num js-line-number" data-line-number="540"></td>
        <td id="LC540" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L541" class="blob-num js-line-number" data-line-number="541"></td>
        <td id="LC541" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md</span></td>
      </tr>
      <tr>
        <td id="L542" class="blob-num js-line-number" data-line-number="542"></td>
        <td id="LC542" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>read_time_machine</span>():  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L543" class="blob-num js-line-number" data-line-number="543"></td>
        <td id="LC543" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Load the time machine book into a list of sentences.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L544" class="blob-num js-line-number" data-line-number="544"></td>
        <td id="LC544" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>with</span> <span class=pl-en>open</span>(<span class=pl-s1>d2l</span>.<span class=pl-en>download</span>(<span class=pl-s>&#39;time_machine&#39;</span>), <span class=pl-s>&#39;r&#39;</span>) <span class=pl-k>as</span> <span class=pl-s1>f</span>:</td>
      </tr>
      <tr>
        <td id="L545" class="blob-num js-line-number" data-line-number="545"></td>
        <td id="LC545" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>lines</span> <span class=pl-c1>=</span> <span class=pl-s1>f</span>.<span class=pl-en>readlines</span>()</td>
      </tr>
      <tr>
        <td id="L546" class="blob-num js-line-number" data-line-number="546"></td>
        <td id="LC546" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> [<span class=pl-s1>re</span>.<span class=pl-en>sub</span>(<span class=pl-s>&#39;[^A-Za-z]+&#39;</span>, <span class=pl-s>&#39; &#39;</span>, <span class=pl-s1>line</span>.<span class=pl-en>strip</span>().<span class=pl-en>lower</span>())</td>
      </tr>
      <tr>
        <td id="L547" class="blob-num js-line-number" data-line-number="547"></td>
        <td id="LC547" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>for</span> <span class=pl-s1>line</span> <span class=pl-c1>in</span> <span class=pl-s1>lines</span>]</td>
      </tr>
      <tr>
        <td id="L548" class="blob-num js-line-number" data-line-number="548"></td>
        <td id="LC548" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L549" class="blob-num js-line-number" data-line-number="549"></td>
        <td id="LC549" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L550" class="blob-num js-line-number" data-line-number="550"></td>
        <td id="LC550" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md</span></td>
      </tr>
      <tr>
        <td id="L551" class="blob-num js-line-number" data-line-number="551"></td>
        <td id="LC551" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>tokenize</span>(<span class=pl-s1>lines</span>, <span class=pl-s1>token</span><span class=pl-c1>=</span><span class=pl-s>&#39;word&#39;</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L552" class="blob-num js-line-number" data-line-number="552"></td>
        <td id="LC552" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Split sentences into word or char tokens.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L553" class="blob-num js-line-number" data-line-number="553"></td>
        <td id="LC553" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-s1>token</span> <span class=pl-c1>==</span> <span class=pl-s>&#39;word&#39;</span>:</td>
      </tr>
      <tr>
        <td id="L554" class="blob-num js-line-number" data-line-number="554"></td>
        <td id="LC554" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> [<span class=pl-s1>line</span>.<span class=pl-en>split</span>(<span class=pl-s>&#39; &#39;</span>) <span class=pl-k>for</span> <span class=pl-s1>line</span> <span class=pl-c1>in</span> <span class=pl-s1>lines</span>]</td>
      </tr>
      <tr>
        <td id="L555" class="blob-num js-line-number" data-line-number="555"></td>
        <td id="LC555" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>elif</span> <span class=pl-s1>token</span> <span class=pl-c1>==</span> <span class=pl-s>&#39;char&#39;</span>:</td>
      </tr>
      <tr>
        <td id="L556" class="blob-num js-line-number" data-line-number="556"></td>
        <td id="LC556" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> [<span class=pl-en>list</span>(<span class=pl-s1>line</span>) <span class=pl-k>for</span> <span class=pl-s1>line</span> <span class=pl-c1>in</span> <span class=pl-s1>lines</span>]</td>
      </tr>
      <tr>
        <td id="L557" class="blob-num js-line-number" data-line-number="557"></td>
        <td id="LC557" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>else</span>:</td>
      </tr>
      <tr>
        <td id="L558" class="blob-num js-line-number" data-line-number="558"></td>
        <td id="LC558" class="blob-code blob-code-inner js-file-line">        <span class=pl-en>print</span>(<span class=pl-s>&#39;ERROR: unknown token type &#39;</span><span class=pl-c1>+</span><span class=pl-s1>token</span>)</td>
      </tr>
      <tr>
        <td id="L559" class="blob-num js-line-number" data-line-number="559"></td>
        <td id="LC559" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L560" class="blob-num js-line-number" data-line-number="560"></td>
        <td id="LC560" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L561" class="blob-num js-line-number" data-line-number="561"></td>
        <td id="LC561" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md</span></td>
      </tr>
      <tr>
        <td id="L562" class="blob-num js-line-number" data-line-number="562"></td>
        <td id="LC562" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>Vocab</span>:  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L563" class="blob-num js-line-number" data-line-number="563"></td>
        <td id="LC563" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>, <span class=pl-s1>tokens</span>, <span class=pl-s1>min_freq</span><span class=pl-c1>=</span><span class=pl-c1>0</span>, <span class=pl-s1>reserved_tokens</span><span class=pl-c1>=</span><span class=pl-c1>None</span>):</td>
      </tr>
      <tr>
        <td id="L564" class="blob-num js-line-number" data-line-number="564"></td>
        <td id="LC564" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>reserved_tokens</span> <span class=pl-c1>is</span> <span class=pl-c1>None</span>:</td>
      </tr>
      <tr>
        <td id="L565" class="blob-num js-line-number" data-line-number="565"></td>
        <td id="LC565" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>reserved_tokens</span> <span class=pl-c1>=</span> []</td>
      </tr>
      <tr>
        <td id="L566" class="blob-num js-line-number" data-line-number="566"></td>
        <td id="LC566" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># Sort according to frequencies</span></td>
      </tr>
      <tr>
        <td id="L567" class="blob-num js-line-number" data-line-number="567"></td>
        <td id="LC567" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>counter</span> <span class=pl-c1>=</span> <span class=pl-en>count_corpus</span>(<span class=pl-s1>tokens</span>)</td>
      </tr>
      <tr>
        <td id="L568" class="blob-num js-line-number" data-line-number="568"></td>
        <td id="LC568" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>token_freqs</span> <span class=pl-c1>=</span> <span class=pl-en>sorted</span>(<span class=pl-s1>counter</span>.<span class=pl-en>items</span>(), <span class=pl-s1>key</span><span class=pl-c1>=</span><span class=pl-k>lambda</span> <span class=pl-s1>x</span>: <span class=pl-s1>x</span>[<span class=pl-c1>0</span>])</td>
      </tr>
      <tr>
        <td id="L569" class="blob-num js-line-number" data-line-number="569"></td>
        <td id="LC569" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>token_freqs</span>.<span class=pl-en>sort</span>(<span class=pl-s1>key</span><span class=pl-c1>=</span><span class=pl-k>lambda</span> <span class=pl-s1>x</span>: <span class=pl-s1>x</span>[<span class=pl-c1>1</span>], <span class=pl-s1>reverse</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L570" class="blob-num js-line-number" data-line-number="570"></td>
        <td id="LC570" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>unk</span>, <span class=pl-s1>uniq_tokens</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span>, [<span class=pl-s>&#39;&lt;unk&gt;&#39;</span>] <span class=pl-c1>+</span> <span class=pl-s1>reserved_tokens</span></td>
      </tr>
      <tr>
        <td id="L571" class="blob-num js-line-number" data-line-number="571"></td>
        <td id="LC571" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>uniq_tokens</span> <span class=pl-c1>+=</span> [<span class=pl-s1>token</span> <span class=pl-k>for</span> <span class=pl-s1>token</span>, <span class=pl-s1>freq</span> <span class=pl-c1>in</span> <span class=pl-s1>self</span>.<span class=pl-s1>token_freqs</span></td>
      </tr>
      <tr>
        <td id="L572" class="blob-num js-line-number" data-line-number="572"></td>
        <td id="LC572" class="blob-code blob-code-inner js-file-line">                        <span class=pl-k>if</span> <span class=pl-s1>freq</span> <span class=pl-c1>&gt;=</span> <span class=pl-s1>min_freq</span> <span class=pl-c1>and</span> <span class=pl-s1>token</span> <span class=pl-c1>not</span> <span class=pl-c1>in</span> <span class=pl-s1>uniq_tokens</span>]</td>
      </tr>
      <tr>
        <td id="L573" class="blob-num js-line-number" data-line-number="573"></td>
        <td id="LC573" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>idx_to_token</span>, <span class=pl-s1>self</span>.<span class=pl-s1>token_to_idx</span> <span class=pl-c1>=</span> [], <span class=pl-en>dict</span>()</td>
      </tr>
      <tr>
        <td id="L574" class="blob-num js-line-number" data-line-number="574"></td>
        <td id="LC574" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-s1>token</span> <span class=pl-c1>in</span> <span class=pl-s1>uniq_tokens</span>:</td>
      </tr>
      <tr>
        <td id="L575" class="blob-num js-line-number" data-line-number="575"></td>
        <td id="LC575" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>self</span>.<span class=pl-s1>idx_to_token</span>.<span class=pl-en>append</span>(<span class=pl-s1>token</span>)</td>
      </tr>
      <tr>
        <td id="L576" class="blob-num js-line-number" data-line-number="576"></td>
        <td id="LC576" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>self</span>.<span class=pl-s1>token_to_idx</span>[<span class=pl-s1>token</span>] <span class=pl-c1>=</span> <span class=pl-en>len</span>(<span class=pl-s1>self</span>.<span class=pl-s1>idx_to_token</span>) <span class=pl-c1>-</span> <span class=pl-c1>1</span></td>
      </tr>
      <tr>
        <td id="L577" class="blob-num js-line-number" data-line-number="577"></td>
        <td id="LC577" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L578" class="blob-num js-line-number" data-line-number="578"></td>
        <td id="LC578" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__len__</span>(<span class=pl-s1>self</span>):</td>
      </tr>
      <tr>
        <td id="L579" class="blob-num js-line-number" data-line-number="579"></td>
        <td id="LC579" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-en>len</span>(<span class=pl-s1>self</span>.<span class=pl-s1>idx_to_token</span>)</td>
      </tr>
      <tr>
        <td id="L580" class="blob-num js-line-number" data-line-number="580"></td>
        <td id="LC580" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L581" class="blob-num js-line-number" data-line-number="581"></td>
        <td id="LC581" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__getitem__</span>(<span class=pl-s1>self</span>, <span class=pl-s1>tokens</span>):</td>
      </tr>
      <tr>
        <td id="L582" class="blob-num js-line-number" data-line-number="582"></td>
        <td id="LC582" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-c1>not</span> <span class=pl-en>isinstance</span>(<span class=pl-s1>tokens</span>, (<span class=pl-s1>list</span>, <span class=pl-s1>tuple</span>)):</td>
      </tr>
      <tr>
        <td id="L583" class="blob-num js-line-number" data-line-number="583"></td>
        <td id="LC583" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>return</span> <span class=pl-s1>self</span>.<span class=pl-s1>token_to_idx</span>.<span class=pl-en>get</span>(<span class=pl-s1>tokens</span>, <span class=pl-s1>self</span>.<span class=pl-s1>unk</span>)</td>
      </tr>
      <tr>
        <td id="L584" class="blob-num js-line-number" data-line-number="584"></td>
        <td id="LC584" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> [<span class=pl-s1>self</span>.<span class=pl-en>__getitem__</span>(<span class=pl-s1>token</span>) <span class=pl-k>for</span> <span class=pl-s1>token</span> <span class=pl-c1>in</span> <span class=pl-s1>tokens</span>]</td>
      </tr>
      <tr>
        <td id="L585" class="blob-num js-line-number" data-line-number="585"></td>
        <td id="LC585" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L586" class="blob-num js-line-number" data-line-number="586"></td>
        <td id="LC586" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>to_tokens</span>(<span class=pl-s1>self</span>, <span class=pl-s1>indices</span>):</td>
      </tr>
      <tr>
        <td id="L587" class="blob-num js-line-number" data-line-number="587"></td>
        <td id="LC587" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-c1>not</span> <span class=pl-en>isinstance</span>(<span class=pl-s1>indices</span>, (<span class=pl-s1>list</span>, <span class=pl-s1>tuple</span>)):</td>
      </tr>
      <tr>
        <td id="L588" class="blob-num js-line-number" data-line-number="588"></td>
        <td id="LC588" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>return</span> <span class=pl-s1>self</span>.<span class=pl-s1>idx_to_token</span>[<span class=pl-s1>indices</span>]</td>
      </tr>
      <tr>
        <td id="L589" class="blob-num js-line-number" data-line-number="589"></td>
        <td id="LC589" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> [<span class=pl-s1>self</span>.<span class=pl-s1>idx_to_token</span>[<span class=pl-s1>index</span>] <span class=pl-k>for</span> <span class=pl-s1>index</span> <span class=pl-c1>in</span> <span class=pl-s1>indices</span>]</td>
      </tr>
      <tr>
        <td id="L590" class="blob-num js-line-number" data-line-number="590"></td>
        <td id="LC590" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L591" class="blob-num js-line-number" data-line-number="591"></td>
        <td id="LC591" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L592" class="blob-num js-line-number" data-line-number="592"></td>
        <td id="LC592" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md</span></td>
      </tr>
      <tr>
        <td id="L593" class="blob-num js-line-number" data-line-number="593"></td>
        <td id="LC593" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>count_corpus</span>(<span class=pl-s1>sentences</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L594" class="blob-num js-line-number" data-line-number="594"></td>
        <td id="LC594" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Flatten a list of token lists into a list of tokens</span></td>
      </tr>
      <tr>
        <td id="L595" class="blob-num js-line-number" data-line-number="595"></td>
        <td id="LC595" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>tokens</span> <span class=pl-c1>=</span> [<span class=pl-s1>tk</span> <span class=pl-k>for</span> <span class=pl-s1>line</span> <span class=pl-c1>in</span> <span class=pl-s1>sentences</span> <span class=pl-k>for</span> <span class=pl-s1>tk</span> <span class=pl-c1>in</span> <span class=pl-s1>line</span>]</td>
      </tr>
      <tr>
        <td id="L596" class="blob-num js-line-number" data-line-number="596"></td>
        <td id="LC596" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>collections</span>.<span class=pl-v>Counter</span>(<span class=pl-s1>tokens</span>)</td>
      </tr>
      <tr>
        <td id="L597" class="blob-num js-line-number" data-line-number="597"></td>
        <td id="LC597" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L598" class="blob-num js-line-number" data-line-number="598"></td>
        <td id="LC598" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L599" class="blob-num js-line-number" data-line-number="599"></td>
        <td id="LC599" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md</span></td>
      </tr>
      <tr>
        <td id="L600" class="blob-num js-line-number" data-line-number="600"></td>
        <td id="LC600" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>load_corpus_time_machine</span>(<span class=pl-s1>max_tokens</span><span class=pl-c1>=</span><span class=pl-c1>-</span><span class=pl-c1>1</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L601" class="blob-num js-line-number" data-line-number="601"></td>
        <td id="LC601" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>lines</span> <span class=pl-c1>=</span> <span class=pl-en>read_time_machine</span>()</td>
      </tr>
      <tr>
        <td id="L602" class="blob-num js-line-number" data-line-number="602"></td>
        <td id="LC602" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>tokens</span> <span class=pl-c1>=</span> <span class=pl-en>tokenize</span>(<span class=pl-s1>lines</span>, <span class=pl-s>&#39;char&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L603" class="blob-num js-line-number" data-line-number="603"></td>
        <td id="LC603" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>vocab</span> <span class=pl-c1>=</span> <span class=pl-v>Vocab</span>(<span class=pl-s1>tokens</span>)</td>
      </tr>
      <tr>
        <td id="L604" class="blob-num js-line-number" data-line-number="604"></td>
        <td id="LC604" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>corpus</span> <span class=pl-c1>=</span> [<span class=pl-s1>vocab</span>[<span class=pl-s1>tk</span>] <span class=pl-k>for</span> <span class=pl-s1>line</span> <span class=pl-c1>in</span> <span class=pl-s1>tokens</span> <span class=pl-k>for</span> <span class=pl-s1>tk</span> <span class=pl-c1>in</span> <span class=pl-s1>line</span>]</td>
      </tr>
      <tr>
        <td id="L605" class="blob-num js-line-number" data-line-number="605"></td>
        <td id="LC605" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-s1>max_tokens</span> <span class=pl-c1>&gt;</span> <span class=pl-c1>0</span>:</td>
      </tr>
      <tr>
        <td id="L606" class="blob-num js-line-number" data-line-number="606"></td>
        <td id="LC606" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>corpus</span> <span class=pl-c1>=</span> <span class=pl-s1>corpus</span>[:<span class=pl-s1>max_tokens</span>]</td>
      </tr>
      <tr>
        <td id="L607" class="blob-num js-line-number" data-line-number="607"></td>
        <td id="LC607" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>corpus</span>, <span class=pl-s1>vocab</span></td>
      </tr>
      <tr>
        <td id="L608" class="blob-num js-line-number" data-line-number="608"></td>
        <td id="LC608" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L609" class="blob-num js-line-number" data-line-number="609"></td>
        <td id="LC609" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L610" class="blob-num js-line-number" data-line-number="610"></td>
        <td id="LC610" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/language-models-and-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L611" class="blob-num js-line-number" data-line-number="611"></td>
        <td id="LC611" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>seq_data_iter_random</span>(<span class=pl-s1>corpus</span>, <span class=pl-s1>batch_size</span>, <span class=pl-s1>num_steps</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L612" class="blob-num js-line-number" data-line-number="612"></td>
        <td id="LC612" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Offset the iterator over the data for uniform starts</span></td>
      </tr>
      <tr>
        <td id="L613" class="blob-num js-line-number" data-line-number="613"></td>
        <td id="LC613" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>corpus</span> <span class=pl-c1>=</span> <span class=pl-s1>corpus</span>[<span class=pl-s1>random</span>.<span class=pl-en>randint</span>(<span class=pl-c1>0</span>, <span class=pl-s1>num_steps</span>):]</td>
      </tr>
      <tr>
        <td id="L614" class="blob-num js-line-number" data-line-number="614"></td>
        <td id="LC614" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Subtract 1 extra since we need to account for label</span></td>
      </tr>
      <tr>
        <td id="L615" class="blob-num js-line-number" data-line-number="615"></td>
        <td id="LC615" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>num_examples</span> <span class=pl-c1>=</span> ((<span class=pl-en>len</span>(<span class=pl-s1>corpus</span>) <span class=pl-c1>-</span> <span class=pl-c1>1</span>) <span class=pl-c1>//</span> <span class=pl-s1>num_steps</span>)</td>
      </tr>
      <tr>
        <td id="L616" class="blob-num js-line-number" data-line-number="616"></td>
        <td id="LC616" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>example_indices</span> <span class=pl-c1>=</span> <span class=pl-en>list</span>(<span class=pl-en>range</span>(<span class=pl-c1>0</span>, <span class=pl-s1>num_examples</span> <span class=pl-c1>*</span> <span class=pl-s1>num_steps</span>, <span class=pl-s1>num_steps</span>))</td>
      </tr>
      <tr>
        <td id="L617" class="blob-num js-line-number" data-line-number="617"></td>
        <td id="LC617" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>random</span>.<span class=pl-en>shuffle</span>(<span class=pl-s1>example_indices</span>)</td>
      </tr>
      <tr>
        <td id="L618" class="blob-num js-line-number" data-line-number="618"></td>
        <td id="LC618" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L619" class="blob-num js-line-number" data-line-number="619"></td>
        <td id="LC619" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>data</span>(<span class=pl-s1>pos</span>):</td>
      </tr>
      <tr>
        <td id="L620" class="blob-num js-line-number" data-line-number="620"></td>
        <td id="LC620" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># This returns a sequence of length `num_steps` starting from `pos`</span></td>
      </tr>
      <tr>
        <td id="L621" class="blob-num js-line-number" data-line-number="621"></td>
        <td id="LC621" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>corpus</span>[<span class=pl-s1>pos</span>: <span class=pl-s1>pos</span> <span class=pl-c1>+</span> <span class=pl-s1>num_steps</span>]</td>
      </tr>
      <tr>
        <td id="L622" class="blob-num js-line-number" data-line-number="622"></td>
        <td id="LC622" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L623" class="blob-num js-line-number" data-line-number="623"></td>
        <td id="LC623" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Discard half empty batches</span></td>
      </tr>
      <tr>
        <td id="L624" class="blob-num js-line-number" data-line-number="624"></td>
        <td id="LC624" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>num_batches</span> <span class=pl-c1>=</span> <span class=pl-s1>num_examples</span> <span class=pl-c1>//</span> <span class=pl-s1>batch_size</span></td>
      </tr>
      <tr>
        <td id="L625" class="blob-num js-line-number" data-line-number="625"></td>
        <td id="LC625" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>i</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-c1>0</span>, <span class=pl-s1>batch_size</span> <span class=pl-c1>*</span> <span class=pl-s1>num_batches</span>, <span class=pl-s1>batch_size</span>):</td>
      </tr>
      <tr>
        <td id="L626" class="blob-num js-line-number" data-line-number="626"></td>
        <td id="LC626" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># `batch_size` indicates the random examples read each time</span></td>
      </tr>
      <tr>
        <td id="L627" class="blob-num js-line-number" data-line-number="627"></td>
        <td id="LC627" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>batch_indices</span> <span class=pl-c1>=</span> <span class=pl-s1>example_indices</span>[<span class=pl-s1>i</span>:(<span class=pl-s1>i</span><span class=pl-c1>+</span><span class=pl-s1>batch_size</span>)]</td>
      </tr>
      <tr>
        <td id="L628" class="blob-num js-line-number" data-line-number="628"></td>
        <td id="LC628" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>X</span> <span class=pl-c1>=</span> [<span class=pl-en>data</span>(<span class=pl-s1>j</span>) <span class=pl-k>for</span> <span class=pl-s1>j</span> <span class=pl-c1>in</span> <span class=pl-s1>batch_indices</span>]</td>
      </tr>
      <tr>
        <td id="L629" class="blob-num js-line-number" data-line-number="629"></td>
        <td id="LC629" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>Y</span> <span class=pl-c1>=</span> [<span class=pl-en>data</span>(<span class=pl-s1>j</span> <span class=pl-c1>+</span> <span class=pl-c1>1</span>) <span class=pl-k>for</span> <span class=pl-s1>j</span> <span class=pl-c1>in</span> <span class=pl-s1>batch_indices</span>]</td>
      </tr>
      <tr>
        <td id="L630" class="blob-num js-line-number" data-line-number="630"></td>
        <td id="LC630" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>yield</span> <span class=pl-s1>d2l</span>.<span class=pl-en>tensor</span>(<span class=pl-v>X</span>), <span class=pl-s1>d2l</span>.<span class=pl-en>tensor</span>(<span class=pl-v>Y</span>)</td>
      </tr>
      <tr>
        <td id="L631" class="blob-num js-line-number" data-line-number="631"></td>
        <td id="LC631" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L632" class="blob-num js-line-number" data-line-number="632"></td>
        <td id="LC632" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L633" class="blob-num js-line-number" data-line-number="633"></td>
        <td id="LC633" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/language-models-and-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L634" class="blob-num js-line-number" data-line-number="634"></td>
        <td id="LC634" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>seq_data_iter_consecutive</span>(<span class=pl-s1>corpus</span>, <span class=pl-s1>batch_size</span>, <span class=pl-s1>num_steps</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L635" class="blob-num js-line-number" data-line-number="635"></td>
        <td id="LC635" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Offset for the iterator over the data for uniform starts</span></td>
      </tr>
      <tr>
        <td id="L636" class="blob-num js-line-number" data-line-number="636"></td>
        <td id="LC636" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>offset</span> <span class=pl-c1>=</span> <span class=pl-s1>random</span>.<span class=pl-en>randint</span>(<span class=pl-c1>0</span>, <span class=pl-s1>num_steps</span>)</td>
      </tr>
      <tr>
        <td id="L637" class="blob-num js-line-number" data-line-number="637"></td>
        <td id="LC637" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Slice out data: ignore `num_steps` and just wrap around</span></td>
      </tr>
      <tr>
        <td id="L638" class="blob-num js-line-number" data-line-number="638"></td>
        <td id="LC638" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>num_indices</span> <span class=pl-c1>=</span> ((<span class=pl-en>len</span>(<span class=pl-s1>corpus</span>) <span class=pl-c1>-</span> <span class=pl-s1>offset</span> <span class=pl-c1>-</span> <span class=pl-c1>1</span>) <span class=pl-c1>//</span> <span class=pl-s1>batch_size</span>) <span class=pl-c1>*</span> <span class=pl-s1>batch_size</span></td>
      </tr>
      <tr>
        <td id="L639" class="blob-num js-line-number" data-line-number="639"></td>
        <td id="LC639" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Xs</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>tensor</span>(<span class=pl-s1>corpus</span>[<span class=pl-s1>offset</span>:<span class=pl-s1>offset</span><span class=pl-c1>+</span><span class=pl-s1>num_indices</span>])</td>
      </tr>
      <tr>
        <td id="L640" class="blob-num js-line-number" data-line-number="640"></td>
        <td id="LC640" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Ys</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>tensor</span>(<span class=pl-s1>corpus</span>[<span class=pl-s1>offset</span><span class=pl-c1>+</span><span class=pl-c1>1</span>:<span class=pl-s1>offset</span><span class=pl-c1>+</span><span class=pl-c1>1</span><span class=pl-c1>+</span><span class=pl-s1>num_indices</span>])</td>
      </tr>
      <tr>
        <td id="L641" class="blob-num js-line-number" data-line-number="641"></td>
        <td id="LC641" class="blob-code blob-code-inner js-file-line">    <span class=pl-v>Xs</span>, <span class=pl-v>Ys</span> <span class=pl-c1>=</span> <span class=pl-v>Xs</span>.<span class=pl-en>reshape</span>(<span class=pl-s1>batch_size</span>, <span class=pl-c1>-</span><span class=pl-c1>1</span>), <span class=pl-v>Ys</span>.<span class=pl-en>reshape</span>(<span class=pl-s1>batch_size</span>, <span class=pl-c1>-</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L642" class="blob-num js-line-number" data-line-number="642"></td>
        <td id="LC642" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>num_batches</span> <span class=pl-c1>=</span> <span class=pl-v>Xs</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>1</span>] <span class=pl-c1>//</span> <span class=pl-s1>num_steps</span></td>
      </tr>
      <tr>
        <td id="L643" class="blob-num js-line-number" data-line-number="643"></td>
        <td id="LC643" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>i</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-c1>0</span>, <span class=pl-s1>num_batches</span> <span class=pl-c1>*</span> <span class=pl-s1>num_steps</span>, <span class=pl-s1>num_steps</span>):</td>
      </tr>
      <tr>
        <td id="L644" class="blob-num js-line-number" data-line-number="644"></td>
        <td id="LC644" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>X</span> <span class=pl-c1>=</span> <span class=pl-v>Xs</span>[:, <span class=pl-s1>i</span>:(<span class=pl-s1>i</span><span class=pl-c1>+</span><span class=pl-s1>num_steps</span>)]</td>
      </tr>
      <tr>
        <td id="L645" class="blob-num js-line-number" data-line-number="645"></td>
        <td id="LC645" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>Y</span> <span class=pl-c1>=</span> <span class=pl-v>Ys</span>[:, <span class=pl-s1>i</span>:(<span class=pl-s1>i</span><span class=pl-c1>+</span><span class=pl-s1>num_steps</span>)]</td>
      </tr>
      <tr>
        <td id="L646" class="blob-num js-line-number" data-line-number="646"></td>
        <td id="LC646" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>yield</span> <span class=pl-v>X</span>, <span class=pl-v>Y</span></td>
      </tr>
      <tr>
        <td id="L647" class="blob-num js-line-number" data-line-number="647"></td>
        <td id="LC647" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L648" class="blob-num js-line-number" data-line-number="648"></td>
        <td id="LC648" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L649" class="blob-num js-line-number" data-line-number="649"></td>
        <td id="LC649" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/language-models-and-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L650" class="blob-num js-line-number" data-line-number="650"></td>
        <td id="LC650" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>SeqDataLoader</span>:  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L651" class="blob-num js-line-number" data-line-number="651"></td>
        <td id="LC651" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;An iterator to load sequence data.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L652" class="blob-num js-line-number" data-line-number="652"></td>
        <td id="LC652" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>, <span class=pl-s1>batch_size</span>, <span class=pl-s1>num_steps</span>, <span class=pl-s1>use_random_iter</span>, <span class=pl-s1>max_tokens</span>):</td>
      </tr>
      <tr>
        <td id="L653" class="blob-num js-line-number" data-line-number="653"></td>
        <td id="LC653" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>use_random_iter</span>:</td>
      </tr>
      <tr>
        <td id="L654" class="blob-num js-line-number" data-line-number="654"></td>
        <td id="LC654" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>self</span>.<span class=pl-s1>data_iter_fn</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-s1>seq_data_iter_random</span></td>
      </tr>
      <tr>
        <td id="L655" class="blob-num js-line-number" data-line-number="655"></td>
        <td id="LC655" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>else</span>:</td>
      </tr>
      <tr>
        <td id="L656" class="blob-num js-line-number" data-line-number="656"></td>
        <td id="LC656" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>self</span>.<span class=pl-s1>data_iter_fn</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-s1>seq_data_iter_consecutive</span></td>
      </tr>
      <tr>
        <td id="L657" class="blob-num js-line-number" data-line-number="657"></td>
        <td id="LC657" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>corpus</span>, <span class=pl-s1>self</span>.<span class=pl-s1>vocab</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>load_corpus_time_machine</span>(<span class=pl-s1>max_tokens</span>)</td>
      </tr>
      <tr>
        <td id="L658" class="blob-num js-line-number" data-line-number="658"></td>
        <td id="LC658" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>batch_size</span>, <span class=pl-s1>self</span>.<span class=pl-s1>num_steps</span> <span class=pl-c1>=</span> <span class=pl-s1>batch_size</span>, <span class=pl-s1>num_steps</span></td>
      </tr>
      <tr>
        <td id="L659" class="blob-num js-line-number" data-line-number="659"></td>
        <td id="LC659" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L660" class="blob-num js-line-number" data-line-number="660"></td>
        <td id="LC660" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__iter__</span>(<span class=pl-s1>self</span>):</td>
      </tr>
      <tr>
        <td id="L661" class="blob-num js-line-number" data-line-number="661"></td>
        <td id="LC661" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>self</span>.<span class=pl-en>data_iter_fn</span>(<span class=pl-s1>self</span>.<span class=pl-s1>corpus</span>, <span class=pl-s1>self</span>.<span class=pl-s1>batch_size</span>, <span class=pl-s1>self</span>.<span class=pl-s1>num_steps</span>)</td>
      </tr>
      <tr>
        <td id="L662" class="blob-num js-line-number" data-line-number="662"></td>
        <td id="LC662" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L663" class="blob-num js-line-number" data-line-number="663"></td>
        <td id="LC663" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L664" class="blob-num js-line-number" data-line-number="664"></td>
        <td id="LC664" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/language-models-and-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L665" class="blob-num js-line-number" data-line-number="665"></td>
        <td id="LC665" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>load_data_time_machine</span>(<span class=pl-s1>batch_size</span>, <span class=pl-s1>num_steps</span>,  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L666" class="blob-num js-line-number" data-line-number="666"></td>
        <td id="LC666" class="blob-code blob-code-inner js-file-line">                           <span class=pl-s1>use_random_iter</span><span class=pl-c1>=</span><span class=pl-c1>False</span>, <span class=pl-s1>max_tokens</span><span class=pl-c1>=</span><span class=pl-c1>10000</span>):</td>
      </tr>
      <tr>
        <td id="L667" class="blob-num js-line-number" data-line-number="667"></td>
        <td id="LC667" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>data_iter</span> <span class=pl-c1>=</span> <span class=pl-v>SeqDataLoader</span>(</td>
      </tr>
      <tr>
        <td id="L668" class="blob-num js-line-number" data-line-number="668"></td>
        <td id="LC668" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>batch_size</span>, <span class=pl-s1>num_steps</span>, <span class=pl-s1>use_random_iter</span>, <span class=pl-s1>max_tokens</span>)</td>
      </tr>
      <tr>
        <td id="L669" class="blob-num js-line-number" data-line-number="669"></td>
        <td id="LC669" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>data_iter</span>, <span class=pl-s1>data_iter</span>.<span class=pl-s1>vocab</span></td>
      </tr>
      <tr>
        <td id="L670" class="blob-num js-line-number" data-line-number="670"></td>
        <td id="LC670" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L671" class="blob-num js-line-number" data-line-number="671"></td>
        <td id="LC671" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L672" class="blob-num js-line-number" data-line-number="672"></td>
        <td id="LC672" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/rnn-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L673" class="blob-num js-line-number" data-line-number="673"></td>
        <td id="LC673" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>RNNModelScratch</span>: <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L674" class="blob-num js-line-number" data-line-number="674"></td>
        <td id="LC674" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;A RNN Model based on scratch implementations.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L675" class="blob-num js-line-number" data-line-number="675"></td>
        <td id="LC675" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>, <span class=pl-s1>vocab_size</span>, <span class=pl-s1>num_hiddens</span>, <span class=pl-s1>device</span>,</td>
      </tr>
      <tr>
        <td id="L676" class="blob-num js-line-number" data-line-number="676"></td>
        <td id="LC676" class="blob-code blob-code-inner js-file-line">                 <span class=pl-s1>get_params</span>, <span class=pl-s1>init_state</span>, <span class=pl-s1>forward</span>):</td>
      </tr>
      <tr>
        <td id="L677" class="blob-num js-line-number" data-line-number="677"></td>
        <td id="LC677" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>vocab_size</span>, <span class=pl-s1>self</span>.<span class=pl-s1>num_hiddens</span> <span class=pl-c1>=</span> <span class=pl-s1>vocab_size</span>, <span class=pl-s1>num_hiddens</span></td>
      </tr>
      <tr>
        <td id="L678" class="blob-num js-line-number" data-line-number="678"></td>
        <td id="LC678" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>params</span> <span class=pl-c1>=</span> <span class=pl-en>get_params</span>(<span class=pl-s1>vocab_size</span>, <span class=pl-s1>num_hiddens</span>, <span class=pl-s1>device</span>)</td>
      </tr>
      <tr>
        <td id="L679" class="blob-num js-line-number" data-line-number="679"></td>
        <td id="LC679" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>init_state</span>, <span class=pl-s1>self</span>.<span class=pl-s1>forward_fn</span> <span class=pl-c1>=</span> <span class=pl-s1>init_state</span>, <span class=pl-s1>forward</span></td>
      </tr>
      <tr>
        <td id="L680" class="blob-num js-line-number" data-line-number="680"></td>
        <td id="LC680" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L681" class="blob-num js-line-number" data-line-number="681"></td>
        <td id="LC681" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__call__</span>(<span class=pl-s1>self</span>, <span class=pl-v>X</span>, <span class=pl-s1>state</span>):</td>
      </tr>
      <tr>
        <td id="L682" class="blob-num js-line-number" data-line-number="682"></td>
        <td id="LC682" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>X</span> <span class=pl-c1>=</span> <span class=pl-v>F</span>.<span class=pl-en>one_hot</span>(<span class=pl-v>X</span>.<span class=pl-v>T</span>.<span class=pl-en>long</span>(), <span class=pl-s1>self</span>.<span class=pl-s1>vocab_size</span>).<span class=pl-en>type</span>(<span class=pl-s1>torch</span>.<span class=pl-s1>float32</span>)</td>
      </tr>
      <tr>
        <td id="L683" class="blob-num js-line-number" data-line-number="683"></td>
        <td id="LC683" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>self</span>.<span class=pl-en>forward_fn</span>(<span class=pl-v>X</span>, <span class=pl-s1>state</span>, <span class=pl-s1>self</span>.<span class=pl-s1>params</span>)</td>
      </tr>
      <tr>
        <td id="L684" class="blob-num js-line-number" data-line-number="684"></td>
        <td id="LC684" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L685" class="blob-num js-line-number" data-line-number="685"></td>
        <td id="LC685" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>begin_state</span>(<span class=pl-s1>self</span>, <span class=pl-s1>batch_size</span>, <span class=pl-s1>device</span>):</td>
      </tr>
      <tr>
        <td id="L686" class="blob-num js-line-number" data-line-number="686"></td>
        <td id="LC686" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>self</span>.<span class=pl-en>init_state</span>(<span class=pl-s1>batch_size</span>, <span class=pl-s1>self</span>.<span class=pl-s1>num_hiddens</span>, <span class=pl-s1>device</span>)</td>
      </tr>
      <tr>
        <td id="L687" class="blob-num js-line-number" data-line-number="687"></td>
        <td id="LC687" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L688" class="blob-num js-line-number" data-line-number="688"></td>
        <td id="LC688" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L689" class="blob-num js-line-number" data-line-number="689"></td>
        <td id="LC689" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/rnn-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L690" class="blob-num js-line-number" data-line-number="690"></td>
        <td id="LC690" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>predict_ch8</span>(<span class=pl-s1>prefix</span>, <span class=pl-s1>num_predicts</span>, <span class=pl-s1>model</span>, <span class=pl-s1>vocab</span>, <span class=pl-s1>device</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L691" class="blob-num js-line-number" data-line-number="691"></td>
        <td id="LC691" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>state</span> <span class=pl-c1>=</span> <span class=pl-s1>model</span>.<span class=pl-en>begin_state</span>(<span class=pl-s1>batch_size</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>device</span><span class=pl-c1>=</span><span class=pl-s1>device</span>)</td>
      </tr>
      <tr>
        <td id="L692" class="blob-num js-line-number" data-line-number="692"></td>
        <td id="LC692" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>outputs</span> <span class=pl-c1>=</span> [<span class=pl-s1>vocab</span>[<span class=pl-s1>prefix</span>[<span class=pl-c1>0</span>]]]</td>
      </tr>
      <tr>
        <td id="L693" class="blob-num js-line-number" data-line-number="693"></td>
        <td id="LC693" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>get_input</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span>: <span class=pl-s1>torch</span>.<span class=pl-en>tensor</span>(</td>
      </tr>
      <tr>
        <td id="L694" class="blob-num js-line-number" data-line-number="694"></td>
        <td id="LC694" class="blob-code blob-code-inner js-file-line">        [<span class=pl-s1>outputs</span>[<span class=pl-c1>-</span><span class=pl-c1>1</span>]], <span class=pl-s1>device</span><span class=pl-c1>=</span><span class=pl-s1>device</span>).<span class=pl-en>reshape</span>(<span class=pl-c1>1</span>, <span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L695" class="blob-num js-line-number" data-line-number="695"></td>
        <td id="LC695" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>y</span> <span class=pl-c1>in</span> <span class=pl-s1>prefix</span>[<span class=pl-c1>1</span>:]:  <span class=pl-c># Warmup state with prefix</span></td>
      </tr>
      <tr>
        <td id="L696" class="blob-num js-line-number" data-line-number="696"></td>
        <td id="LC696" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>_</span>, <span class=pl-s1>state</span> <span class=pl-c1>=</span> <span class=pl-en>model</span>(<span class=pl-en>get_input</span>(), <span class=pl-s1>state</span>)</td>
      </tr>
      <tr>
        <td id="L697" class="blob-num js-line-number" data-line-number="697"></td>
        <td id="LC697" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>outputs</span>.<span class=pl-en>append</span>(<span class=pl-s1>vocab</span>[<span class=pl-s1>y</span>])</td>
      </tr>
      <tr>
        <td id="L698" class="blob-num js-line-number" data-line-number="698"></td>
        <td id="LC698" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>_</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-s1>num_predicts</span>):  <span class=pl-c># Predict num_predicts steps</span></td>
      </tr>
      <tr>
        <td id="L699" class="blob-num js-line-number" data-line-number="699"></td>
        <td id="LC699" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>Y</span>, <span class=pl-s1>state</span> <span class=pl-c1>=</span> <span class=pl-en>model</span>(<span class=pl-en>get_input</span>(), <span class=pl-s1>state</span>)</td>
      </tr>
      <tr>
        <td id="L700" class="blob-num js-line-number" data-line-number="700"></td>
        <td id="LC700" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>outputs</span>.<span class=pl-en>append</span>(<span class=pl-en>int</span>(<span class=pl-v>Y</span>.<span class=pl-en>argmax</span>(<span class=pl-s1>dim</span><span class=pl-c1>=</span><span class=pl-c1>1</span>).<span class=pl-en>reshape</span>(<span class=pl-c1>1</span>)))</td>
      </tr>
      <tr>
        <td id="L701" class="blob-num js-line-number" data-line-number="701"></td>
        <td id="LC701" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s>&#39;&#39;</span>.<span class=pl-en>join</span>([<span class=pl-s1>vocab</span>.<span class=pl-s1>idx_to_token</span>[<span class=pl-s1>i</span>] <span class=pl-k>for</span> <span class=pl-s1>i</span> <span class=pl-c1>in</span> <span class=pl-s1>outputs</span>])</td>
      </tr>
      <tr>
        <td id="L702" class="blob-num js-line-number" data-line-number="702"></td>
        <td id="LC702" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L703" class="blob-num js-line-number" data-line-number="703"></td>
        <td id="LC703" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L704" class="blob-num js-line-number" data-line-number="704"></td>
        <td id="LC704" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/rnn-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L705" class="blob-num js-line-number" data-line-number="705"></td>
        <td id="LC705" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>grad_clipping</span>(<span class=pl-s1>model</span>, <span class=pl-s1>theta</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L706" class="blob-num js-line-number" data-line-number="706"></td>
        <td id="LC706" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-en>isinstance</span>(<span class=pl-s1>model</span>, <span class=pl-s1>nn</span>.<span class=pl-v>Module</span>):</td>
      </tr>
      <tr>
        <td id="L707" class="blob-num js-line-number" data-line-number="707"></td>
        <td id="LC707" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>params</span> <span class=pl-c1>=</span> [<span class=pl-s1>p</span> <span class=pl-k>for</span> <span class=pl-s1>p</span> <span class=pl-c1>in</span> <span class=pl-s1>model</span>.<span class=pl-en>parameters</span>() <span class=pl-k>if</span> <span class=pl-s1>p</span>.<span class=pl-s1>requires_grad</span>]</td>
      </tr>
      <tr>
        <td id="L708" class="blob-num js-line-number" data-line-number="708"></td>
        <td id="LC708" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>else</span>:</td>
      </tr>
      <tr>
        <td id="L709" class="blob-num js-line-number" data-line-number="709"></td>
        <td id="LC709" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>params</span> <span class=pl-c1>=</span> <span class=pl-s1>model</span>.<span class=pl-s1>params</span></td>
      </tr>
      <tr>
        <td id="L710" class="blob-num js-line-number" data-line-number="710"></td>
        <td id="LC710" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>norm</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-en>sqrt</span>(<span class=pl-en>sum</span>(<span class=pl-s1>torch</span>.<span class=pl-en>sum</span>((<span class=pl-s1>p</span>.<span class=pl-s1>grad</span> <span class=pl-c1>**</span> <span class=pl-c1>2</span>)) <span class=pl-k>for</span> <span class=pl-s1>p</span> <span class=pl-c1>in</span> <span class=pl-s1>params</span>))</td>
      </tr>
      <tr>
        <td id="L711" class="blob-num js-line-number" data-line-number="711"></td>
        <td id="LC711" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-s1>norm</span> <span class=pl-c1>&gt;</span> <span class=pl-s1>theta</span>:</td>
      </tr>
      <tr>
        <td id="L712" class="blob-num js-line-number" data-line-number="712"></td>
        <td id="LC712" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-s1>param</span> <span class=pl-c1>in</span> <span class=pl-s1>params</span>:</td>
      </tr>
      <tr>
        <td id="L713" class="blob-num js-line-number" data-line-number="713"></td>
        <td id="LC713" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>param</span>.<span class=pl-s1>grad</span>[:] <span class=pl-c1>*=</span> <span class=pl-s1>theta</span> <span class=pl-c1>/</span> <span class=pl-s1>norm</span></td>
      </tr>
      <tr>
        <td id="L714" class="blob-num js-line-number" data-line-number="714"></td>
        <td id="LC714" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L715" class="blob-num js-line-number" data-line-number="715"></td>
        <td id="LC715" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L716" class="blob-num js-line-number" data-line-number="716"></td>
        <td id="LC716" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/rnn-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L717" class="blob-num js-line-number" data-line-number="717"></td>
        <td id="LC717" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>train_epoch_ch8</span>(<span class=pl-s1>model</span>, <span class=pl-s1>train_iter</span>, <span class=pl-s1>loss</span>, <span class=pl-s1>updater</span>, <span class=pl-s1>device</span>,  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L718" class="blob-num js-line-number" data-line-number="718"></td>
        <td id="LC718" class="blob-code blob-code-inner js-file-line">                    <span class=pl-s1>use_random_iter</span>):</td>
      </tr>
      <tr>
        <td id="L719" class="blob-num js-line-number" data-line-number="719"></td>
        <td id="LC719" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>state</span>, <span class=pl-s1>timer</span> <span class=pl-c1>=</span> <span class=pl-c1>None</span>, <span class=pl-s1>d2l</span>.<span class=pl-v>Timer</span>()</td>
      </tr>
      <tr>
        <td id="L720" class="blob-num js-line-number" data-line-number="720"></td>
        <td id="LC720" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>metric</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Accumulator</span>(<span class=pl-c1>2</span>)  <span class=pl-c># loss_sum, num_examples</span></td>
      </tr>
      <tr>
        <td id="L721" class="blob-num js-line-number" data-line-number="721"></td>
        <td id="LC721" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-v>X</span>, <span class=pl-v>Y</span> <span class=pl-c1>in</span> <span class=pl-s1>train_iter</span>:</td>
      </tr>
      <tr>
        <td id="L722" class="blob-num js-line-number" data-line-number="722"></td>
        <td id="LC722" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>state</span> <span class=pl-c1>is</span> <span class=pl-c1>None</span> <span class=pl-c1>or</span> <span class=pl-s1>use_random_iter</span>:</td>
      </tr>
      <tr>
        <td id="L723" class="blob-num js-line-number" data-line-number="723"></td>
        <td id="LC723" class="blob-code blob-code-inner js-file-line">            <span class=pl-c># Initialize state when either it is the first iteration or</span></td>
      </tr>
      <tr>
        <td id="L724" class="blob-num js-line-number" data-line-number="724"></td>
        <td id="LC724" class="blob-code blob-code-inner js-file-line">            <span class=pl-c># using random sampling.</span></td>
      </tr>
      <tr>
        <td id="L725" class="blob-num js-line-number" data-line-number="725"></td>
        <td id="LC725" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>state</span> <span class=pl-c1>=</span> <span class=pl-s1>model</span>.<span class=pl-en>begin_state</span>(<span class=pl-s1>batch_size</span><span class=pl-c1>=</span><span class=pl-v>X</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>0</span>], <span class=pl-s1>device</span><span class=pl-c1>=</span><span class=pl-s1>device</span>)</td>
      </tr>
      <tr>
        <td id="L726" class="blob-num js-line-number" data-line-number="726"></td>
        <td id="LC726" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>else</span>:</td>
      </tr>
      <tr>
        <td id="L727" class="blob-num js-line-number" data-line-number="727"></td>
        <td id="LC727" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>for</span> <span class=pl-s1>s</span> <span class=pl-c1>in</span> <span class=pl-s1>state</span>:</td>
      </tr>
      <tr>
        <td id="L728" class="blob-num js-line-number" data-line-number="728"></td>
        <td id="LC728" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>s</span>.<span class=pl-en>detach_</span>()</td>
      </tr>
      <tr>
        <td id="L729" class="blob-num js-line-number" data-line-number="729"></td>
        <td id="LC729" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-v>Y</span>.<span class=pl-v>T</span>.<span class=pl-en>reshape</span>(<span class=pl-c1>-</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L730" class="blob-num js-line-number" data-line-number="730"></td>
        <td id="LC730" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>X</span>, <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-v>X</span>.<span class=pl-en>to</span>(<span class=pl-s1>device</span>), <span class=pl-s1>y</span>.<span class=pl-en>to</span>(<span class=pl-s1>device</span>)</td>
      </tr>
      <tr>
        <td id="L731" class="blob-num js-line-number" data-line-number="731"></td>
        <td id="LC731" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>py</span>, <span class=pl-s1>state</span> <span class=pl-c1>=</span> <span class=pl-en>model</span>(<span class=pl-v>X</span>, <span class=pl-s1>state</span>)</td>
      </tr>
      <tr>
        <td id="L732" class="blob-num js-line-number" data-line-number="732"></td>
        <td id="LC732" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>l</span> <span class=pl-c1>=</span> <span class=pl-en>loss</span>(<span class=pl-s1>py</span>, <span class=pl-s1>y</span>.<span class=pl-en>long</span>()).<span class=pl-en>mean</span>()</td>
      </tr>
      <tr>
        <td id="L733" class="blob-num js-line-number" data-line-number="733"></td>
        <td id="LC733" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-en>isinstance</span>(<span class=pl-s1>updater</span>, <span class=pl-s1>torch</span>.<span class=pl-s1>optim</span>.<span class=pl-v>Optimizer</span>):</td>
      </tr>
      <tr>
        <td id="L734" class="blob-num js-line-number" data-line-number="734"></td>
        <td id="LC734" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>updater</span>.<span class=pl-en>zero_grad</span>()</td>
      </tr>
      <tr>
        <td id="L735" class="blob-num js-line-number" data-line-number="735"></td>
        <td id="LC735" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>l</span>.<span class=pl-en>backward</span>()</td>
      </tr>
      <tr>
        <td id="L736" class="blob-num js-line-number" data-line-number="736"></td>
        <td id="LC736" class="blob-code blob-code-inner js-file-line">            <span class=pl-en>grad_clipping</span>(<span class=pl-s1>model</span>, <span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L737" class="blob-num js-line-number" data-line-number="737"></td>
        <td id="LC737" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>updater</span>.<span class=pl-en>step</span>()</td>
      </tr>
      <tr>
        <td id="L738" class="blob-num js-line-number" data-line-number="738"></td>
        <td id="LC738" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>else</span>:</td>
      </tr>
      <tr>
        <td id="L739" class="blob-num js-line-number" data-line-number="739"></td>
        <td id="LC739" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>l</span>.<span class=pl-en>backward</span>()</td>
      </tr>
      <tr>
        <td id="L740" class="blob-num js-line-number" data-line-number="740"></td>
        <td id="LC740" class="blob-code blob-code-inner js-file-line">            <span class=pl-en>grad_clipping</span>(<span class=pl-s1>model</span>, <span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L741" class="blob-num js-line-number" data-line-number="741"></td>
        <td id="LC741" class="blob-code blob-code-inner js-file-line">            <span class=pl-en>updater</span>(<span class=pl-s1>batch_size</span><span class=pl-c1>=</span><span class=pl-c1>1</span>)  <span class=pl-c># Since used mean already</span></td>
      </tr>
      <tr>
        <td id="L742" class="blob-num js-line-number" data-line-number="742"></td>
        <td id="LC742" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>metric</span>.<span class=pl-en>add</span>(<span class=pl-s1>l</span> <span class=pl-c1>*</span> <span class=pl-s1>d2l</span>.<span class=pl-en>size</span>(<span class=pl-s1>y</span>), <span class=pl-s1>d2l</span>.<span class=pl-en>size</span>(<span class=pl-s1>y</span>))</td>
      </tr>
      <tr>
        <td id="L743" class="blob-num js-line-number" data-line-number="743"></td>
        <td id="LC743" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>math</span>.<span class=pl-en>exp</span>(<span class=pl-s1>metric</span>[<span class=pl-c1>0</span>] <span class=pl-c1>/</span> <span class=pl-s1>metric</span>[<span class=pl-c1>1</span>]), <span class=pl-s1>metric</span>[<span class=pl-c1>1</span>] <span class=pl-c1>/</span> <span class=pl-s1>timer</span>.<span class=pl-en>stop</span>()</td>
      </tr>
      <tr>
        <td id="L744" class="blob-num js-line-number" data-line-number="744"></td>
        <td id="LC744" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L745" class="blob-num js-line-number" data-line-number="745"></td>
        <td id="LC745" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L746" class="blob-num js-line-number" data-line-number="746"></td>
        <td id="LC746" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-neural-networks/rnn-scratch.md</span></td>
      </tr>
      <tr>
        <td id="L747" class="blob-num js-line-number" data-line-number="747"></td>
        <td id="LC747" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>train_ch8</span>(<span class=pl-s1>model</span>, <span class=pl-s1>train_iter</span>, <span class=pl-s1>vocab</span>, <span class=pl-s1>lr</span>, <span class=pl-s1>num_epochs</span>, <span class=pl-s1>device</span>,</td>
      </tr>
      <tr>
        <td id="L748" class="blob-num js-line-number" data-line-number="748"></td>
        <td id="LC748" class="blob-code blob-code-inner js-file-line">              <span class=pl-s1>use_random_iter</span><span class=pl-c1>=</span><span class=pl-c1>False</span>):</td>
      </tr>
      <tr>
        <td id="L749" class="blob-num js-line-number" data-line-number="749"></td>
        <td id="LC749" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Initialize</span></td>
      </tr>
      <tr>
        <td id="L750" class="blob-num js-line-number" data-line-number="750"></td>
        <td id="LC750" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>loss</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>CrossEntropyLoss</span>()</td>
      </tr>
      <tr>
        <td id="L751" class="blob-num js-line-number" data-line-number="751"></td>
        <td id="LC751" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>animator</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Animator</span>(<span class=pl-s1>xlabel</span><span class=pl-c1>=</span><span class=pl-s>&#39;epoch&#39;</span>, <span class=pl-s1>ylabel</span><span class=pl-c1>=</span><span class=pl-s>&#39;perplexity&#39;</span>,</td>
      </tr>
      <tr>
        <td id="L752" class="blob-num js-line-number" data-line-number="752"></td>
        <td id="LC752" class="blob-code blob-code-inner js-file-line">                            <span class=pl-s1>legend</span><span class=pl-c1>=</span>[<span class=pl-s>&#39;train&#39;</span>], <span class=pl-s1>xlim</span><span class=pl-c1>=</span>[<span class=pl-c1>1</span>, <span class=pl-s1>num_epochs</span>])</td>
      </tr>
      <tr>
        <td id="L753" class="blob-num js-line-number" data-line-number="753"></td>
        <td id="LC753" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-en>isinstance</span>(<span class=pl-s1>model</span>, <span class=pl-s1>nn</span>.<span class=pl-v>Module</span>):</td>
      </tr>
      <tr>
        <td id="L754" class="blob-num js-line-number" data-line-number="754"></td>
        <td id="LC754" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>trainer</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>optim</span>.<span class=pl-v>SGD</span>(<span class=pl-s1>model</span>.<span class=pl-en>parameters</span>(), <span class=pl-s1>lr</span>)</td>
      </tr>
      <tr>
        <td id="L755" class="blob-num js-line-number" data-line-number="755"></td>
        <td id="LC755" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>updater</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span> <span class=pl-s1>batch_size</span>: <span class=pl-s1>trainer</span>.<span class=pl-en>step</span>()</td>
      </tr>
      <tr>
        <td id="L756" class="blob-num js-line-number" data-line-number="756"></td>
        <td id="LC756" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>else</span>:</td>
      </tr>
      <tr>
        <td id="L757" class="blob-num js-line-number" data-line-number="757"></td>
        <td id="LC757" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>updater</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span> <span class=pl-s1>batch_size</span>: <span class=pl-s1>d2l</span>.<span class=pl-en>sgd</span>(<span class=pl-s1>model</span>.<span class=pl-s1>params</span>, <span class=pl-s1>lr</span>, <span class=pl-s1>batch_size</span>)</td>
      </tr>
      <tr>
        <td id="L758" class="blob-num js-line-number" data-line-number="758"></td>
        <td id="LC758" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>predict</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span> <span class=pl-s1>prefix</span>: <span class=pl-en>predict_ch8</span>(<span class=pl-s1>prefix</span>, <span class=pl-c1>50</span>, <span class=pl-s1>model</span>, <span class=pl-s1>vocab</span>, <span class=pl-s1>device</span>)</td>
      </tr>
      <tr>
        <td id="L759" class="blob-num js-line-number" data-line-number="759"></td>
        <td id="LC759" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Train and check the progress.</span></td>
      </tr>
      <tr>
        <td id="L760" class="blob-num js-line-number" data-line-number="760"></td>
        <td id="LC760" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>epoch</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-s1>num_epochs</span>):</td>
      </tr>
      <tr>
        <td id="L761" class="blob-num js-line-number" data-line-number="761"></td>
        <td id="LC761" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>ppl</span>, <span class=pl-s1>speed</span> <span class=pl-c1>=</span> <span class=pl-en>train_epoch_ch8</span>(</td>
      </tr>
      <tr>
        <td id="L762" class="blob-num js-line-number" data-line-number="762"></td>
        <td id="LC762" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>model</span>, <span class=pl-s1>train_iter</span>, <span class=pl-s1>loss</span>, <span class=pl-s1>updater</span>, <span class=pl-s1>device</span>, <span class=pl-s1>use_random_iter</span>)</td>
      </tr>
      <tr>
        <td id="L763" class="blob-num js-line-number" data-line-number="763"></td>
        <td id="LC763" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>epoch</span> <span class=pl-c1>%</span> <span class=pl-c1>10</span> <span class=pl-c1>==</span> <span class=pl-c1>0</span>:</td>
      </tr>
      <tr>
        <td id="L764" class="blob-num js-line-number" data-line-number="764"></td>
        <td id="LC764" class="blob-code blob-code-inner js-file-line">            <span class=pl-en>print</span>(<span class=pl-en>predict</span>(<span class=pl-s>&#39;time traveller&#39;</span>))</td>
      </tr>
      <tr>
        <td id="L765" class="blob-num js-line-number" data-line-number="765"></td>
        <td id="LC765" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>animator</span>.<span class=pl-en>add</span>(<span class=pl-s1>epoch</span><span class=pl-c1>+</span><span class=pl-c1>1</span>, [<span class=pl-s1>ppl</span>])</td>
      </tr>
      <tr>
        <td id="L766" class="blob-num js-line-number" data-line-number="766"></td>
        <td id="LC766" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>f&#39;perplexity <span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>ppl</span>:.1f<span class=pl-kos>}</span></span>, <span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>speed</span>:.1f<span class=pl-kos>}</span></span> tokens/sec on <span class=pl-s1><span class=pl-kos>{</span><span class=pl-en>str</span>(<span class=pl-s1>device</span>)<span class=pl-kos>}</span></span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L767" class="blob-num js-line-number" data-line-number="767"></td>
        <td id="LC767" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-en>predict</span>(<span class=pl-s>&#39;time traveller&#39;</span>))</td>
      </tr>
      <tr>
        <td id="L768" class="blob-num js-line-number" data-line-number="768"></td>
        <td id="LC768" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-en>predict</span>(<span class=pl-s>&#39;traveller&#39;</span>))</td>
      </tr>
      <tr>
        <td id="L769" class="blob-num js-line-number" data-line-number="769"></td>
        <td id="LC769" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L770" class="blob-num js-line-number" data-line-number="770"></td>
        <td id="LC770" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L771" class="blob-num js-line-number" data-line-number="771"></td>
        <td id="LC771" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/machine-translation-and-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L772" class="blob-num js-line-number" data-line-number="772"></td>
        <td id="LC772" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>d2l</span>.<span class=pl-v>DATA_HUB</span>[<span class=pl-s>&#39;fra-eng&#39;</span>] <span class=pl-c1>=</span> (<span class=pl-s1>d2l</span>.<span class=pl-v>DATA_URL</span> <span class=pl-c1>+</span> <span class=pl-s>&#39;fra-eng.zip&#39;</span>,</td>
      </tr>
      <tr>
        <td id="L773" class="blob-num js-line-number" data-line-number="773"></td>
        <td id="LC773" class="blob-code blob-code-inner js-file-line">                           <span class=pl-s>&#39;94646ad1522d915e7b0f9296181140edcf86a4f5&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L774" class="blob-num js-line-number" data-line-number="774"></td>
        <td id="LC774" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L775" class="blob-num js-line-number" data-line-number="775"></td>
        <td id="LC775" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L776" class="blob-num js-line-number" data-line-number="776"></td>
        <td id="LC776" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/machine-translation-and-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L777" class="blob-num js-line-number" data-line-number="777"></td>
        <td id="LC777" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>read_data_nmt</span>():</td>
      </tr>
      <tr>
        <td id="L778" class="blob-num js-line-number" data-line-number="778"></td>
        <td id="LC778" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>data_dir</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>download_extract</span>(<span class=pl-s>&#39;fra-eng&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L779" class="blob-num js-line-number" data-line-number="779"></td>
        <td id="LC779" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>with</span> <span class=pl-en>open</span>(<span class=pl-s1>os</span>.<span class=pl-s1>path</span>.<span class=pl-en>join</span>(<span class=pl-s1>data_dir</span>, <span class=pl-s>&#39;fra.txt&#39;</span>), <span class=pl-s>&#39;r&#39;</span>) <span class=pl-k>as</span> <span class=pl-s1>f</span>:</td>
      </tr>
      <tr>
        <td id="L780" class="blob-num js-line-number" data-line-number="780"></td>
        <td id="LC780" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>f</span>.<span class=pl-en>read</span>()</td>
      </tr>
      <tr>
        <td id="L781" class="blob-num js-line-number" data-line-number="781"></td>
        <td id="LC781" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L782" class="blob-num js-line-number" data-line-number="782"></td>
        <td id="LC782" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L783" class="blob-num js-line-number" data-line-number="783"></td>
        <td id="LC783" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/machine-translation-and-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L784" class="blob-num js-line-number" data-line-number="784"></td>
        <td id="LC784" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>preprocess_nmt</span>(<span class=pl-s1>text</span>):</td>
      </tr>
      <tr>
        <td id="L785" class="blob-num js-line-number" data-line-number="785"></td>
        <td id="LC785" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>no_space</span>(<span class=pl-s1>char</span>, <span class=pl-s1>prev_char</span>):</td>
      </tr>
      <tr>
        <td id="L786" class="blob-num js-line-number" data-line-number="786"></td>
        <td id="LC786" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>char</span> <span class=pl-c1>in</span> <span class=pl-en>set</span>(<span class=pl-s>&#39;,.!&#39;</span>) <span class=pl-c1>and</span> <span class=pl-s1>prev_char</span> <span class=pl-c1>!=</span> <span class=pl-s>&#39; &#39;</span></td>
      </tr>
      <tr>
        <td id="L787" class="blob-num js-line-number" data-line-number="787"></td>
        <td id="LC787" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L788" class="blob-num js-line-number" data-line-number="788"></td>
        <td id="LC788" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>text</span> <span class=pl-c1>=</span> <span class=pl-s1>text</span>.<span class=pl-en>replace</span>(<span class=pl-s>&#39;<span class=pl-cce>\u202f</span>&#39;</span>, <span class=pl-s>&#39; &#39;</span>).<span class=pl-en>replace</span>(<span class=pl-s>&#39;<span class=pl-cce>\xa0</span>&#39;</span>, <span class=pl-s>&#39; &#39;</span>).<span class=pl-en>lower</span>()</td>
      </tr>
      <tr>
        <td id="L789" class="blob-num js-line-number" data-line-number="789"></td>
        <td id="LC789" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>out</span> <span class=pl-c1>=</span> [<span class=pl-s>&#39; &#39;</span> <span class=pl-c1>+</span> <span class=pl-s1>char</span> <span class=pl-k>if</span> <span class=pl-s1>i</span> <span class=pl-c1>&gt;</span> <span class=pl-c1>0</span> <span class=pl-c1>and</span> <span class=pl-en>no_space</span>(<span class=pl-s1>char</span>, <span class=pl-s1>text</span>[<span class=pl-s1>i</span><span class=pl-c1>-</span><span class=pl-c1>1</span>]) <span class=pl-k>else</span> <span class=pl-s1>char</span></td>
      </tr>
      <tr>
        <td id="L790" class="blob-num js-line-number" data-line-number="790"></td>
        <td id="LC790" class="blob-code blob-code-inner js-file-line">           <span class=pl-k>for</span> <span class=pl-s1>i</span>, <span class=pl-s1>char</span> <span class=pl-c1>in</span> <span class=pl-en>enumerate</span>(<span class=pl-s1>text</span>)]</td>
      </tr>
      <tr>
        <td id="L791" class="blob-num js-line-number" data-line-number="791"></td>
        <td id="LC791" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s>&#39;&#39;</span>.<span class=pl-en>join</span>(<span class=pl-s1>out</span>)</td>
      </tr>
      <tr>
        <td id="L792" class="blob-num js-line-number" data-line-number="792"></td>
        <td id="LC792" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L793" class="blob-num js-line-number" data-line-number="793"></td>
        <td id="LC793" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L794" class="blob-num js-line-number" data-line-number="794"></td>
        <td id="LC794" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/machine-translation-and-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L795" class="blob-num js-line-number" data-line-number="795"></td>
        <td id="LC795" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>tokenize_nmt</span>(<span class=pl-s1>text</span>, <span class=pl-s1>num_examples</span><span class=pl-c1>=</span><span class=pl-c1>None</span>):</td>
      </tr>
      <tr>
        <td id="L796" class="blob-num js-line-number" data-line-number="796"></td>
        <td id="LC796" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>source</span>, <span class=pl-s1>target</span> <span class=pl-c1>=</span> [], []</td>
      </tr>
      <tr>
        <td id="L797" class="blob-num js-line-number" data-line-number="797"></td>
        <td id="LC797" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>i</span>, <span class=pl-s1>line</span> <span class=pl-c1>in</span> <span class=pl-en>enumerate</span>(<span class=pl-s1>text</span>.<span class=pl-en>split</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span>&#39;</span>)):</td>
      </tr>
      <tr>
        <td id="L798" class="blob-num js-line-number" data-line-number="798"></td>
        <td id="LC798" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>num_examples</span> <span class=pl-c1>and</span> <span class=pl-s1>i</span> <span class=pl-c1>&gt;</span> <span class=pl-s1>num_examples</span>:</td>
      </tr>
      <tr>
        <td id="L799" class="blob-num js-line-number" data-line-number="799"></td>
        <td id="LC799" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>break</span></td>
      </tr>
      <tr>
        <td id="L800" class="blob-num js-line-number" data-line-number="800"></td>
        <td id="LC800" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>parts</span> <span class=pl-c1>=</span> <span class=pl-s1>line</span>.<span class=pl-en>split</span>(<span class=pl-s>&#39;<span class=pl-cce>\t</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L801" class="blob-num js-line-number" data-line-number="801"></td>
        <td id="LC801" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-en>len</span>(<span class=pl-s1>parts</span>) <span class=pl-c1>==</span> <span class=pl-c1>2</span>:</td>
      </tr>
      <tr>
        <td id="L802" class="blob-num js-line-number" data-line-number="802"></td>
        <td id="LC802" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>source</span>.<span class=pl-en>append</span>(<span class=pl-s1>parts</span>[<span class=pl-c1>0</span>].<span class=pl-en>split</span>(<span class=pl-s>&#39; &#39;</span>))</td>
      </tr>
      <tr>
        <td id="L803" class="blob-num js-line-number" data-line-number="803"></td>
        <td id="LC803" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>target</span>.<span class=pl-en>append</span>(<span class=pl-s1>parts</span>[<span class=pl-c1>1</span>].<span class=pl-en>split</span>(<span class=pl-s>&#39; &#39;</span>))</td>
      </tr>
      <tr>
        <td id="L804" class="blob-num js-line-number" data-line-number="804"></td>
        <td id="LC804" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>source</span>, <span class=pl-s1>target</span></td>
      </tr>
      <tr>
        <td id="L805" class="blob-num js-line-number" data-line-number="805"></td>
        <td id="LC805" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L806" class="blob-num js-line-number" data-line-number="806"></td>
        <td id="LC806" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L807" class="blob-num js-line-number" data-line-number="807"></td>
        <td id="LC807" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/machine-translation-and-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L808" class="blob-num js-line-number" data-line-number="808"></td>
        <td id="LC808" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>truncate_pad</span>(<span class=pl-s1>line</span>, <span class=pl-s1>num_steps</span>, <span class=pl-s1>padding_token</span>):</td>
      </tr>
      <tr>
        <td id="L809" class="blob-num js-line-number" data-line-number="809"></td>
        <td id="LC809" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-en>len</span>(<span class=pl-s1>line</span>) <span class=pl-c1>&gt;</span> <span class=pl-s1>num_steps</span>:</td>
      </tr>
      <tr>
        <td id="L810" class="blob-num js-line-number" data-line-number="810"></td>
        <td id="LC810" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>line</span>[:<span class=pl-s1>num_steps</span>]  <span class=pl-c># Trim</span></td>
      </tr>
      <tr>
        <td id="L811" class="blob-num js-line-number" data-line-number="811"></td>
        <td id="LC811" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>line</span> <span class=pl-c1>+</span> [<span class=pl-s1>padding_token</span>] <span class=pl-c1>*</span> (<span class=pl-s1>num_steps</span> <span class=pl-c1>-</span> <span class=pl-en>len</span>(<span class=pl-s1>line</span>))  <span class=pl-c># Pad</span></td>
      </tr>
      <tr>
        <td id="L812" class="blob-num js-line-number" data-line-number="812"></td>
        <td id="LC812" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L813" class="blob-num js-line-number" data-line-number="813"></td>
        <td id="LC813" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L814" class="blob-num js-line-number" data-line-number="814"></td>
        <td id="LC814" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/machine-translation-and-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L815" class="blob-num js-line-number" data-line-number="815"></td>
        <td id="LC815" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>build_array</span>(<span class=pl-s1>lines</span>, <span class=pl-s1>vocab</span>, <span class=pl-s1>num_steps</span>, <span class=pl-s1>is_source</span>):</td>
      </tr>
      <tr>
        <td id="L816" class="blob-num js-line-number" data-line-number="816"></td>
        <td id="LC816" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>lines</span> <span class=pl-c1>=</span> [<span class=pl-s1>vocab</span>[<span class=pl-s1>l</span>] <span class=pl-k>for</span> <span class=pl-s1>l</span> <span class=pl-c1>in</span> <span class=pl-s1>lines</span>]</td>
      </tr>
      <tr>
        <td id="L817" class="blob-num js-line-number" data-line-number="817"></td>
        <td id="LC817" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-c1>not</span> <span class=pl-s1>is_source</span>:</td>
      </tr>
      <tr>
        <td id="L818" class="blob-num js-line-number" data-line-number="818"></td>
        <td id="LC818" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>lines</span> <span class=pl-c1>=</span> [[<span class=pl-s1>vocab</span>[<span class=pl-s>&#39;&lt;bos&gt;&#39;</span>]] <span class=pl-c1>+</span> <span class=pl-s1>l</span> <span class=pl-c1>+</span> [<span class=pl-s1>vocab</span>[<span class=pl-s>&#39;&lt;eos&gt;&#39;</span>]] <span class=pl-k>for</span> <span class=pl-s1>l</span> <span class=pl-c1>in</span> <span class=pl-s1>lines</span>]</td>
      </tr>
      <tr>
        <td id="L819" class="blob-num js-line-number" data-line-number="819"></td>
        <td id="LC819" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>array</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-en>tensor</span>([<span class=pl-en>truncate_pad</span>(</td>
      </tr>
      <tr>
        <td id="L820" class="blob-num js-line-number" data-line-number="820"></td>
        <td id="LC820" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>l</span>, <span class=pl-s1>num_steps</span>, <span class=pl-s1>vocab</span>[<span class=pl-s>&#39;&lt;pad&gt;&#39;</span>]) <span class=pl-k>for</span> <span class=pl-s1>l</span> <span class=pl-c1>in</span> <span class=pl-s1>lines</span>])</td>
      </tr>
      <tr>
        <td id="L821" class="blob-num js-line-number" data-line-number="821"></td>
        <td id="LC821" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>valid_len</span> <span class=pl-c1>=</span> (<span class=pl-s1>array</span> <span class=pl-c1>!=</span> <span class=pl-s1>vocab</span>[<span class=pl-s>&#39;&lt;pad&gt;&#39;</span>]).<span class=pl-en>sum</span>(<span class=pl-s1>dim</span><span class=pl-c1>=</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L822" class="blob-num js-line-number" data-line-number="822"></td>
        <td id="LC822" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>array</span>, <span class=pl-s1>valid_len</span></td>
      </tr>
      <tr>
        <td id="L823" class="blob-num js-line-number" data-line-number="823"></td>
        <td id="LC823" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L824" class="blob-num js-line-number" data-line-number="824"></td>
        <td id="LC824" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L825" class="blob-num js-line-number" data-line-number="825"></td>
        <td id="LC825" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/machine-translation-and-dataset.md</span></td>
      </tr>
      <tr>
        <td id="L826" class="blob-num js-line-number" data-line-number="826"></td>
        <td id="LC826" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>load_data_nmt</span>(<span class=pl-s1>batch_size</span>, <span class=pl-s1>num_steps</span>, <span class=pl-s1>num_examples</span><span class=pl-c1>=</span><span class=pl-c1>1000</span>):</td>
      </tr>
      <tr>
        <td id="L827" class="blob-num js-line-number" data-line-number="827"></td>
        <td id="LC827" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>text</span> <span class=pl-c1>=</span> <span class=pl-en>preprocess_nmt</span>(<span class=pl-en>read_data_nmt</span>())</td>
      </tr>
      <tr>
        <td id="L828" class="blob-num js-line-number" data-line-number="828"></td>
        <td id="LC828" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>source</span>, <span class=pl-s1>target</span> <span class=pl-c1>=</span> <span class=pl-en>tokenize_nmt</span>(<span class=pl-s1>text</span>, <span class=pl-s1>num_examples</span>)</td>
      </tr>
      <tr>
        <td id="L829" class="blob-num js-line-number" data-line-number="829"></td>
        <td id="LC829" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>src_vocab</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Vocab</span>(<span class=pl-s1>source</span>, <span class=pl-s1>min_freq</span><span class=pl-c1>=</span><span class=pl-c1>3</span>, </td>
      </tr>
      <tr>
        <td id="L830" class="blob-num js-line-number" data-line-number="830"></td>
        <td id="LC830" class="blob-code blob-code-inner js-file-line">                          <span class=pl-s1>reserved_tokens</span><span class=pl-c1>=</span>[<span class=pl-s>&#39;&lt;pad&gt;&#39;</span>, <span class=pl-s>&#39;&lt;bos&gt;&#39;</span>, <span class=pl-s>&#39;&lt;eos&gt;&#39;</span>])</td>
      </tr>
      <tr>
        <td id="L831" class="blob-num js-line-number" data-line-number="831"></td>
        <td id="LC831" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>tgt_vocab</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Vocab</span>(<span class=pl-s1>target</span>, <span class=pl-s1>min_freq</span><span class=pl-c1>=</span><span class=pl-c1>3</span>, </td>
      </tr>
      <tr>
        <td id="L832" class="blob-num js-line-number" data-line-number="832"></td>
        <td id="LC832" class="blob-code blob-code-inner js-file-line">                          <span class=pl-s1>reserved_tokens</span><span class=pl-c1>=</span>[<span class=pl-s>&#39;&lt;pad&gt;&#39;</span>, <span class=pl-s>&#39;&lt;bos&gt;&#39;</span>, <span class=pl-s>&#39;&lt;eos&gt;&#39;</span>])</td>
      </tr>
      <tr>
        <td id="L833" class="blob-num js-line-number" data-line-number="833"></td>
        <td id="LC833" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>src_array</span>, <span class=pl-s1>src_valid_len</span> <span class=pl-c1>=</span> <span class=pl-en>build_array</span>(</td>
      </tr>
      <tr>
        <td id="L834" class="blob-num js-line-number" data-line-number="834"></td>
        <td id="LC834" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>source</span>, <span class=pl-s1>src_vocab</span>, <span class=pl-s1>num_steps</span>, <span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L835" class="blob-num js-line-number" data-line-number="835"></td>
        <td id="LC835" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>tgt_array</span>, <span class=pl-s1>tgt_valid_len</span> <span class=pl-c1>=</span> <span class=pl-en>build_array</span>(</td>
      </tr>
      <tr>
        <td id="L836" class="blob-num js-line-number" data-line-number="836"></td>
        <td id="LC836" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>target</span>, <span class=pl-s1>tgt_vocab</span>, <span class=pl-s1>num_steps</span>, <span class=pl-c1>False</span>)</td>
      </tr>
      <tr>
        <td id="L837" class="blob-num js-line-number" data-line-number="837"></td>
        <td id="LC837" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>data_arrays</span> <span class=pl-c1>=</span> (<span class=pl-s1>src_array</span>, <span class=pl-s1>src_valid_len</span>, <span class=pl-s1>tgt_array</span>, <span class=pl-s1>tgt_valid_len</span>)</td>
      </tr>
      <tr>
        <td id="L838" class="blob-num js-line-number" data-line-number="838"></td>
        <td id="LC838" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>data_iter</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>load_array</span>(<span class=pl-s1>data_arrays</span>, <span class=pl-s1>batch_size</span>)</td>
      </tr>
      <tr>
        <td id="L839" class="blob-num js-line-number" data-line-number="839"></td>
        <td id="LC839" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>src_vocab</span>, <span class=pl-s1>tgt_vocab</span>, <span class=pl-s1>data_iter</span></td>
      </tr>
      <tr>
        <td id="L840" class="blob-num js-line-number" data-line-number="840"></td>
        <td id="LC840" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L841" class="blob-num js-line-number" data-line-number="841"></td>
        <td id="LC841" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L842" class="blob-num js-line-number" data-line-number="842"></td>
        <td id="LC842" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/encoder-decoder.md</span></td>
      </tr>
      <tr>
        <td id="L843" class="blob-num js-line-number" data-line-number="843"></td>
        <td id="LC843" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>Encoder</span>(<span class=pl-s1>nn</span>.<span class=pl-v>Module</span>):</td>
      </tr>
      <tr>
        <td id="L844" class="blob-num js-line-number" data-line-number="844"></td>
        <td id="LC844" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;The base encoder interface for the encoder-decoder architecture.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L845" class="blob-num js-line-number" data-line-number="845"></td>
        <td id="LC845" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>):</td>
      </tr>
      <tr>
        <td id="L846" class="blob-num js-line-number" data-line-number="846"></td>
        <td id="LC846" class="blob-code blob-code-inner js-file-line">        <span class=pl-en>super</span>(<span class=pl-v>Encoder</span>, <span class=pl-s1>self</span>).<span class=pl-en>__init__</span>(<span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L847" class="blob-num js-line-number" data-line-number="847"></td>
        <td id="LC847" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L848" class="blob-num js-line-number" data-line-number="848"></td>
        <td id="LC848" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>forward</span>(<span class=pl-s1>self</span>, <span class=pl-v>X</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>):</td>
      </tr>
      <tr>
        <td id="L849" class="blob-num js-line-number" data-line-number="849"></td>
        <td id="LC849" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>raise</span> <span class=pl-v>NotImplementedError</span></td>
      </tr>
      <tr>
        <td id="L850" class="blob-num js-line-number" data-line-number="850"></td>
        <td id="LC850" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L851" class="blob-num js-line-number" data-line-number="851"></td>
        <td id="LC851" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L852" class="blob-num js-line-number" data-line-number="852"></td>
        <td id="LC852" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/encoder-decoder.md</span></td>
      </tr>
      <tr>
        <td id="L853" class="blob-num js-line-number" data-line-number="853"></td>
        <td id="LC853" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>Decoder</span>(<span class=pl-s1>nn</span>.<span class=pl-v>Module</span>):</td>
      </tr>
      <tr>
        <td id="L854" class="blob-num js-line-number" data-line-number="854"></td>
        <td id="LC854" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;The base decoder interface for the encoder-decoder architecture.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L855" class="blob-num js-line-number" data-line-number="855"></td>
        <td id="LC855" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>):</td>
      </tr>
      <tr>
        <td id="L856" class="blob-num js-line-number" data-line-number="856"></td>
        <td id="LC856" class="blob-code blob-code-inner js-file-line">        <span class=pl-en>super</span>(<span class=pl-v>Decoder</span>, <span class=pl-s1>self</span>).<span class=pl-en>__init__</span>(<span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L857" class="blob-num js-line-number" data-line-number="857"></td>
        <td id="LC857" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L858" class="blob-num js-line-number" data-line-number="858"></td>
        <td id="LC858" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>init_state</span>(<span class=pl-s1>self</span>, <span class=pl-s1>enc_outputs</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>):</td>
      </tr>
      <tr>
        <td id="L859" class="blob-num js-line-number" data-line-number="859"></td>
        <td id="LC859" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>raise</span> <span class=pl-v>NotImplementedError</span></td>
      </tr>
      <tr>
        <td id="L860" class="blob-num js-line-number" data-line-number="860"></td>
        <td id="LC860" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L861" class="blob-num js-line-number" data-line-number="861"></td>
        <td id="LC861" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>forward</span>(<span class=pl-s1>self</span>, <span class=pl-v>X</span>, <span class=pl-s1>state</span>):</td>
      </tr>
      <tr>
        <td id="L862" class="blob-num js-line-number" data-line-number="862"></td>
        <td id="LC862" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>raise</span> <span class=pl-v>NotImplementedError</span></td>
      </tr>
      <tr>
        <td id="L863" class="blob-num js-line-number" data-line-number="863"></td>
        <td id="LC863" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L864" class="blob-num js-line-number" data-line-number="864"></td>
        <td id="LC864" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L865" class="blob-num js-line-number" data-line-number="865"></td>
        <td id="LC865" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/encoder-decoder.md</span></td>
      </tr>
      <tr>
        <td id="L866" class="blob-num js-line-number" data-line-number="866"></td>
        <td id="LC866" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>EncoderDecoder</span>(<span class=pl-s1>nn</span>.<span class=pl-v>Module</span>):</td>
      </tr>
      <tr>
        <td id="L867" class="blob-num js-line-number" data-line-number="867"></td>
        <td id="LC867" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;The base class for the encoder-decoder architecture.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L868" class="blob-num js-line-number" data-line-number="868"></td>
        <td id="LC868" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>, <span class=pl-s1>encoder</span>, <span class=pl-s1>decoder</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>):</td>
      </tr>
      <tr>
        <td id="L869" class="blob-num js-line-number" data-line-number="869"></td>
        <td id="LC869" class="blob-code blob-code-inner js-file-line">        <span class=pl-en>super</span>(<span class=pl-v>EncoderDecoder</span>, <span class=pl-s1>self</span>).<span class=pl-en>__init__</span>(<span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L870" class="blob-num js-line-number" data-line-number="870"></td>
        <td id="LC870" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>encoder</span> <span class=pl-c1>=</span> <span class=pl-s1>encoder</span></td>
      </tr>
      <tr>
        <td id="L871" class="blob-num js-line-number" data-line-number="871"></td>
        <td id="LC871" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>decoder</span> <span class=pl-c1>=</span> <span class=pl-s1>decoder</span></td>
      </tr>
      <tr>
        <td id="L872" class="blob-num js-line-number" data-line-number="872"></td>
        <td id="LC872" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L873" class="blob-num js-line-number" data-line-number="873"></td>
        <td id="LC873" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>forward</span>(<span class=pl-s1>self</span>, <span class=pl-s1>enc_X</span>, <span class=pl-s1>dec_X</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>):</td>
      </tr>
      <tr>
        <td id="L874" class="blob-num js-line-number" data-line-number="874"></td>
        <td id="LC874" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>enc_outputs</span> <span class=pl-c1>=</span> <span class=pl-s1>self</span>.<span class=pl-en>encoder</span>(<span class=pl-s1>enc_X</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>)</td>
      </tr>
      <tr>
        <td id="L875" class="blob-num js-line-number" data-line-number="875"></td>
        <td id="LC875" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>dec_state</span> <span class=pl-c1>=</span> <span class=pl-s1>self</span>.<span class=pl-s1>decoder</span>.<span class=pl-en>init_state</span>(<span class=pl-s1>enc_outputs</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>)</td>
      </tr>
      <tr>
        <td id="L876" class="blob-num js-line-number" data-line-number="876"></td>
        <td id="LC876" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>self</span>.<span class=pl-en>decoder</span>(<span class=pl-s1>dec_X</span>, <span class=pl-s1>dec_state</span>)</td>
      </tr>
      <tr>
        <td id="L877" class="blob-num js-line-number" data-line-number="877"></td>
        <td id="LC877" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L878" class="blob-num js-line-number" data-line-number="878"></td>
        <td id="LC878" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L879" class="blob-num js-line-number" data-line-number="879"></td>
        <td id="LC879" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/seq2seq.md</span></td>
      </tr>
      <tr>
        <td id="L880" class="blob-num js-line-number" data-line-number="880"></td>
        <td id="LC880" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>Seq2SeqEncoder</span>(<span class=pl-s1>d2l</span>.<span class=pl-v>Encoder</span>):</td>
      </tr>
      <tr>
        <td id="L881" class="blob-num js-line-number" data-line-number="881"></td>
        <td id="LC881" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>, <span class=pl-s1>vocab_size</span>, <span class=pl-s1>embed_size</span>, <span class=pl-s1>num_hiddens</span>, <span class=pl-s1>num_layers</span>,</td>
      </tr>
      <tr>
        <td id="L882" class="blob-num js-line-number" data-line-number="882"></td>
        <td id="LC882" class="blob-code blob-code-inner js-file-line">                 <span class=pl-s1>dropout</span><span class=pl-c1>=</span><span class=pl-c1>0</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>):</td>
      </tr>
      <tr>
        <td id="L883" class="blob-num js-line-number" data-line-number="883"></td>
        <td id="LC883" class="blob-code blob-code-inner js-file-line">        <span class=pl-en>super</span>(<span class=pl-v>Seq2SeqEncoder</span>, <span class=pl-s1>self</span>).<span class=pl-en>__init__</span>(<span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L884" class="blob-num js-line-number" data-line-number="884"></td>
        <td id="LC884" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>embedding</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>Embedding</span>(<span class=pl-s1>vocab_size</span>, <span class=pl-s1>embed_size</span>)</td>
      </tr>
      <tr>
        <td id="L885" class="blob-num js-line-number" data-line-number="885"></td>
        <td id="LC885" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>rnn</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>LSTM</span>(<span class=pl-s1>embed_size</span>, <span class=pl-s1>num_hiddens</span>, <span class=pl-s1>num_layers</span>, <span class=pl-s1>dropout</span><span class=pl-c1>=</span><span class=pl-s1>dropout</span>)</td>
      </tr>
      <tr>
        <td id="L886" class="blob-num js-line-number" data-line-number="886"></td>
        <td id="LC886" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L887" class="blob-num js-line-number" data-line-number="887"></td>
        <td id="LC887" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>forward</span>(<span class=pl-s1>self</span>, <span class=pl-v>X</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>):</td>
      </tr>
      <tr>
        <td id="L888" class="blob-num js-line-number" data-line-number="888"></td>
        <td id="LC888" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>X</span> <span class=pl-c1>=</span> <span class=pl-s1>self</span>.<span class=pl-en>embedding</span>(<span class=pl-v>X</span>)  <span class=pl-c># X shape: (batch_size, seq_len, embed_size)</span></td>
      </tr>
      <tr>
        <td id="L889" class="blob-num js-line-number" data-line-number="889"></td>
        <td id="LC889" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># RNN needs first axes to be timestep, i.e., seq_len</span></td>
      </tr>
      <tr>
        <td id="L890" class="blob-num js-line-number" data-line-number="890"></td>
        <td id="LC890" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>X</span> <span class=pl-c1>=</span> <span class=pl-v>X</span>.<span class=pl-en>permute</span>(<span class=pl-c1>1</span>, <span class=pl-c1>0</span>, <span class=pl-c1>2</span>)</td>
      </tr>
      <tr>
        <td id="L891" class="blob-num js-line-number" data-line-number="891"></td>
        <td id="LC891" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>out</span>, <span class=pl-s1>state</span> <span class=pl-c1>=</span> <span class=pl-s1>self</span>.<span class=pl-en>rnn</span>(<span class=pl-v>X</span>) <span class=pl-c># When state is not mentioned, it defaults to zeros</span></td>
      </tr>
      <tr>
        <td id="L892" class="blob-num js-line-number" data-line-number="892"></td>
        <td id="LC892" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># out shape: (seq_len, batch_size, num_hiddens)</span></td>
      </tr>
      <tr>
        <td id="L893" class="blob-num js-line-number" data-line-number="893"></td>
        <td id="LC893" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># state shape: (num_layers, batch_size, num_hiddens),</span></td>
      </tr>
      <tr>
        <td id="L894" class="blob-num js-line-number" data-line-number="894"></td>
        <td id="LC894" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># where &quot;state&quot; contains the hidden state and the memory cell</span></td>
      </tr>
      <tr>
        <td id="L895" class="blob-num js-line-number" data-line-number="895"></td>
        <td id="LC895" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>out</span>, <span class=pl-s1>state</span></td>
      </tr>
      <tr>
        <td id="L896" class="blob-num js-line-number" data-line-number="896"></td>
        <td id="LC896" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L897" class="blob-num js-line-number" data-line-number="897"></td>
        <td id="LC897" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L898" class="blob-num js-line-number" data-line-number="898"></td>
        <td id="LC898" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/seq2seq.md</span></td>
      </tr>
      <tr>
        <td id="L899" class="blob-num js-line-number" data-line-number="899"></td>
        <td id="LC899" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>Seq2SeqDecoder</span>(<span class=pl-s1>d2l</span>.<span class=pl-v>Decoder</span>):</td>
      </tr>
      <tr>
        <td id="L900" class="blob-num js-line-number" data-line-number="900"></td>
        <td id="LC900" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>, <span class=pl-s1>vocab_size</span>, <span class=pl-s1>embed_size</span>, <span class=pl-s1>num_hiddens</span>, <span class=pl-s1>num_layers</span>,</td>
      </tr>
      <tr>
        <td id="L901" class="blob-num js-line-number" data-line-number="901"></td>
        <td id="LC901" class="blob-code blob-code-inner js-file-line">                 <span class=pl-s1>dropout</span><span class=pl-c1>=</span><span class=pl-c1>0</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>):</td>
      </tr>
      <tr>
        <td id="L902" class="blob-num js-line-number" data-line-number="902"></td>
        <td id="LC902" class="blob-code blob-code-inner js-file-line">        <span class=pl-en>super</span>(<span class=pl-v>Seq2SeqDecoder</span>, <span class=pl-s1>self</span>).<span class=pl-en>__init__</span>(<span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L903" class="blob-num js-line-number" data-line-number="903"></td>
        <td id="LC903" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>embedding</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>Embedding</span>(<span class=pl-s1>vocab_size</span>, <span class=pl-s1>embed_size</span>)</td>
      </tr>
      <tr>
        <td id="L904" class="blob-num js-line-number" data-line-number="904"></td>
        <td id="LC904" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>rnn</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>LSTM</span>(<span class=pl-s1>embed_size</span>, <span class=pl-s1>num_hiddens</span>, <span class=pl-s1>num_layers</span>, <span class=pl-s1>dropout</span><span class=pl-c1>=</span><span class=pl-s1>dropout</span>)</td>
      </tr>
      <tr>
        <td id="L905" class="blob-num js-line-number" data-line-number="905"></td>
        <td id="LC905" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>dense</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>Linear</span>(<span class=pl-s1>num_hiddens</span>, <span class=pl-s1>vocab_size</span>)</td>
      </tr>
      <tr>
        <td id="L906" class="blob-num js-line-number" data-line-number="906"></td>
        <td id="LC906" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L907" class="blob-num js-line-number" data-line-number="907"></td>
        <td id="LC907" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>init_state</span>(<span class=pl-s1>self</span>, <span class=pl-s1>enc_outputs</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>):</td>
      </tr>
      <tr>
        <td id="L908" class="blob-num js-line-number" data-line-number="908"></td>
        <td id="LC908" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>enc_outputs</span>[<span class=pl-c1>1</span>]</td>
      </tr>
      <tr>
        <td id="L909" class="blob-num js-line-number" data-line-number="909"></td>
        <td id="LC909" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L910" class="blob-num js-line-number" data-line-number="910"></td>
        <td id="LC910" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>forward</span>(<span class=pl-s1>self</span>, <span class=pl-v>X</span>, <span class=pl-s1>state</span>):</td>
      </tr>
      <tr>
        <td id="L911" class="blob-num js-line-number" data-line-number="911"></td>
        <td id="LC911" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>X</span> <span class=pl-c1>=</span> <span class=pl-s1>self</span>.<span class=pl-en>embedding</span>(<span class=pl-v>X</span>).<span class=pl-en>permute</span>(<span class=pl-c1>1</span>, <span class=pl-c1>0</span>, <span class=pl-c1>2</span>)</td>
      </tr>
      <tr>
        <td id="L912" class="blob-num js-line-number" data-line-number="912"></td>
        <td id="LC912" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>out</span>, <span class=pl-s1>state</span> <span class=pl-c1>=</span> <span class=pl-s1>self</span>.<span class=pl-en>rnn</span>(<span class=pl-v>X</span>, <span class=pl-s1>state</span>)</td>
      </tr>
      <tr>
        <td id="L913" class="blob-num js-line-number" data-line-number="913"></td>
        <td id="LC913" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># Make the batch to be the first dimension to simplify loss computation</span></td>
      </tr>
      <tr>
        <td id="L914" class="blob-num js-line-number" data-line-number="914"></td>
        <td id="LC914" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>out</span> <span class=pl-c1>=</span> <span class=pl-s1>self</span>.<span class=pl-en>dense</span>(<span class=pl-s1>out</span>).<span class=pl-en>permute</span>(<span class=pl-c1>1</span>, <span class=pl-c1>0</span>, <span class=pl-c1>2</span>)</td>
      </tr>
      <tr>
        <td id="L915" class="blob-num js-line-number" data-line-number="915"></td>
        <td id="LC915" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>out</span>, <span class=pl-s1>state</span></td>
      </tr>
      <tr>
        <td id="L916" class="blob-num js-line-number" data-line-number="916"></td>
        <td id="LC916" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L917" class="blob-num js-line-number" data-line-number="917"></td>
        <td id="LC917" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L918" class="blob-num js-line-number" data-line-number="918"></td>
        <td id="LC918" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/seq2seq.md</span></td>
      </tr>
      <tr>
        <td id="L919" class="blob-num js-line-number" data-line-number="919"></td>
        <td id="LC919" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>sequence_mask</span>(<span class=pl-v>X</span>, <span class=pl-s1>valid_len</span>, <span class=pl-s1>value</span><span class=pl-c1>=</span><span class=pl-c1>0</span>):</td>
      </tr>
      <tr>
        <td id="L920" class="blob-num js-line-number" data-line-number="920"></td>
        <td id="LC920" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>output</span> <span class=pl-c1>=</span> <span class=pl-v>X</span>.<span class=pl-en>clone</span>()</td>
      </tr>
      <tr>
        <td id="L921" class="blob-num js-line-number" data-line-number="921"></td>
        <td id="LC921" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>count</span>, <span class=pl-s1>matrix</span> <span class=pl-c1>in</span> <span class=pl-en>enumerate</span>(<span class=pl-s1>output</span>):</td>
      </tr>
      <tr>
        <td id="L922" class="blob-num js-line-number" data-line-number="922"></td>
        <td id="LC922" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>matrix</span>[<span class=pl-en>int</span>(<span class=pl-s1>valid_len</span>[<span class=pl-s1>count</span>]):]<span class=pl-c1>=</span><span class=pl-s1>value</span></td>
      </tr>
      <tr>
        <td id="L923" class="blob-num js-line-number" data-line-number="923"></td>
        <td id="LC923" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>output</span></td>
      </tr>
      <tr>
        <td id="L924" class="blob-num js-line-number" data-line-number="924"></td>
        <td id="LC924" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L925" class="blob-num js-line-number" data-line-number="925"></td>
        <td id="LC925" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L926" class="blob-num js-line-number" data-line-number="926"></td>
        <td id="LC926" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/seq2seq.md</span></td>
      </tr>
      <tr>
        <td id="L927" class="blob-num js-line-number" data-line-number="927"></td>
        <td id="LC927" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>MaskedSoftmaxCELoss</span>(<span class=pl-s1>nn</span>.<span class=pl-v>CrossEntropyLoss</span>):</td>
      </tr>
      <tr>
        <td id="L928" class="blob-num js-line-number" data-line-number="928"></td>
        <td id="LC928" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># pred shape: (batch_size, seq_len, vocab_size)</span></td>
      </tr>
      <tr>
        <td id="L929" class="blob-num js-line-number" data-line-number="929"></td>
        <td id="LC929" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># label shape: (batch_size, seq_len)</span></td>
      </tr>
      <tr>
        <td id="L930" class="blob-num js-line-number" data-line-number="930"></td>
        <td id="LC930" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># valid_len shape: (batch_size, )</span></td>
      </tr>
      <tr>
        <td id="L931" class="blob-num js-line-number" data-line-number="931"></td>
        <td id="LC931" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>forward</span>(<span class=pl-s1>self</span>, <span class=pl-s1>pred</span>, <span class=pl-s1>label</span>, <span class=pl-s1>valid_len</span>):</td>
      </tr>
      <tr>
        <td id="L932" class="blob-num js-line-number" data-line-number="932"></td>
        <td id="LC932" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>weights</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-en>ones_like</span>(<span class=pl-s1>label</span>)</td>
      </tr>
      <tr>
        <td id="L933" class="blob-num js-line-number" data-line-number="933"></td>
        <td id="LC933" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>weights</span> <span class=pl-c1>=</span> <span class=pl-en>sequence_mask</span>(<span class=pl-s1>weights</span>, <span class=pl-s1>valid_len</span>)</td>
      </tr>
      <tr>
        <td id="L934" class="blob-num js-line-number" data-line-number="934"></td>
        <td id="LC934" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>reduction</span><span class=pl-c1>=</span><span class=pl-s>&#39;none&#39;</span></td>
      </tr>
      <tr>
        <td id="L935" class="blob-num js-line-number" data-line-number="935"></td>
        <td id="LC935" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>unweighted_loss</span> <span class=pl-c1>=</span> <span class=pl-en>super</span>(<span class=pl-v>MaskedSoftmaxCELoss</span>, <span class=pl-s1>self</span>).<span class=pl-en>forward</span>(<span class=pl-s1>pred</span>.<span class=pl-en>permute</span>(<span class=pl-c1>0</span>,<span class=pl-c1>2</span>,<span class=pl-c1>1</span>), <span class=pl-s1>label</span>)</td>
      </tr>
      <tr>
        <td id="L936" class="blob-num js-line-number" data-line-number="936"></td>
        <td id="LC936" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>weighted_loss</span> <span class=pl-c1>=</span> (<span class=pl-s1>unweighted_loss</span><span class=pl-c1>*</span><span class=pl-s1>weights</span>).<span class=pl-en>mean</span>(<span class=pl-s1>dim</span><span class=pl-c1>=</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L937" class="blob-num js-line-number" data-line-number="937"></td>
        <td id="LC937" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>weighted_loss</span></td>
      </tr>
      <tr>
        <td id="L938" class="blob-num js-line-number" data-line-number="938"></td>
        <td id="LC938" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L939" class="blob-num js-line-number" data-line-number="939"></td>
        <td id="LC939" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L940" class="blob-num js-line-number" data-line-number="940"></td>
        <td id="LC940" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/seq2seq.md</span></td>
      </tr>
      <tr>
        <td id="L941" class="blob-num js-line-number" data-line-number="941"></td>
        <td id="LC941" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>train_s2s_ch9</span>(<span class=pl-s1>model</span>, <span class=pl-s1>data_iter</span>, <span class=pl-s1>lr</span>, <span class=pl-s1>num_epochs</span>, <span class=pl-s1>device</span>):</td>
      </tr>
      <tr>
        <td id="L942" class="blob-num js-line-number" data-line-number="942"></td>
        <td id="LC942" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>xavier_init_weights</span>(<span class=pl-s1>m</span>):</td>
      </tr>
      <tr>
        <td id="L943" class="blob-num js-line-number" data-line-number="943"></td>
        <td id="LC943" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-en>type</span>(<span class=pl-s1>m</span>) <span class=pl-c1>==</span> <span class=pl-s1>nn</span>.<span class=pl-v>Linear</span>:</td>
      </tr>
      <tr>
        <td id="L944" class="blob-num js-line-number" data-line-number="944"></td>
        <td id="LC944" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>torch</span>.<span class=pl-s1>nn</span>.<span class=pl-s1>init</span>.<span class=pl-en>xavier_uniform_</span>(<span class=pl-s1>m</span>.<span class=pl-s1>weight</span>)</td>
      </tr>
      <tr>
        <td id="L945" class="blob-num js-line-number" data-line-number="945"></td>
        <td id="LC945" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-en>type</span>(<span class=pl-s1>m</span>) <span class=pl-c1>==</span> <span class=pl-s1>nn</span>.<span class=pl-v>LSTM</span>:</td>
      </tr>
      <tr>
        <td id="L946" class="blob-num js-line-number" data-line-number="946"></td>
        <td id="LC946" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>for</span> <span class=pl-s1>param</span> <span class=pl-c1>in</span> <span class=pl-s1>m</span>.<span class=pl-s1>_flat_weights_names</span>:</td>
      </tr>
      <tr>
        <td id="L947" class="blob-num js-line-number" data-line-number="947"></td>
        <td id="LC947" class="blob-code blob-code-inner js-file-line">                <span class=pl-k>if</span> <span class=pl-s>&quot;weight&quot;</span> <span class=pl-c1>in</span> <span class=pl-s1>param</span>:</td>
      </tr>
      <tr>
        <td id="L948" class="blob-num js-line-number" data-line-number="948"></td>
        <td id="LC948" class="blob-code blob-code-inner js-file-line">                    <span class=pl-s1>torch</span>.<span class=pl-s1>nn</span>.<span class=pl-s1>init</span>.<span class=pl-en>xavier_uniform_</span>(<span class=pl-s1>m</span>.<span class=pl-s1>_parameters</span>[<span class=pl-s1>param</span>])</td>
      </tr>
      <tr>
        <td id="L949" class="blob-num js-line-number" data-line-number="949"></td>
        <td id="LC949" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>model</span>.<span class=pl-en>apply</span>(<span class=pl-s1>xavier_init_weights</span>)</td>
      </tr>
      <tr>
        <td id="L950" class="blob-num js-line-number" data-line-number="950"></td>
        <td id="LC950" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>model</span>.<span class=pl-en>to</span>(<span class=pl-s1>device</span>)</td>
      </tr>
      <tr>
        <td id="L951" class="blob-num js-line-number" data-line-number="951"></td>
        <td id="LC951" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>optimizer</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>optim</span>.<span class=pl-v>Adam</span>(<span class=pl-s1>model</span>.<span class=pl-en>parameters</span>(), <span class=pl-s1>lr</span><span class=pl-c1>=</span><span class=pl-s1>lr</span>)</td>
      </tr>
      <tr>
        <td id="L952" class="blob-num js-line-number" data-line-number="952"></td>
        <td id="LC952" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>loss</span> <span class=pl-c1>=</span> <span class=pl-v>MaskedSoftmaxCELoss</span>()</td>
      </tr>
      <tr>
        <td id="L953" class="blob-num js-line-number" data-line-number="953"></td>
        <td id="LC953" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>model</span>.<span class=pl-en>train</span>()</td>
      </tr>
      <tr>
        <td id="L954" class="blob-num js-line-number" data-line-number="954"></td>
        <td id="LC954" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>animator</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Animator</span>(<span class=pl-s1>xlabel</span><span class=pl-c1>=</span><span class=pl-s>&#39;epoch&#39;</span>, <span class=pl-s1>ylabel</span><span class=pl-c1>=</span><span class=pl-s>&#39;loss&#39;</span>,</td>
      </tr>
      <tr>
        <td id="L955" class="blob-num js-line-number" data-line-number="955"></td>
        <td id="LC955" class="blob-code blob-code-inner js-file-line">                            <span class=pl-s1>xlim</span><span class=pl-c1>=</span>[<span class=pl-c1>1</span>, <span class=pl-s1>num_epochs</span>], <span class=pl-s1>ylim</span><span class=pl-c1>=</span>[<span class=pl-c1>0</span>, <span class=pl-c1>0.25</span>])</td>
      </tr>
      <tr>
        <td id="L956" class="blob-num js-line-number" data-line-number="956"></td>
        <td id="LC956" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>epoch</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-c1>1</span>, <span class=pl-s1>num_epochs</span> <span class=pl-c1>+</span> <span class=pl-c1>1</span>):</td>
      </tr>
      <tr>
        <td id="L957" class="blob-num js-line-number" data-line-number="957"></td>
        <td id="LC957" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>timer</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Timer</span>()</td>
      </tr>
      <tr>
        <td id="L958" class="blob-num js-line-number" data-line-number="958"></td>
        <td id="LC958" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>metric</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Accumulator</span>(<span class=pl-c1>2</span>)  <span class=pl-c># loss_sum, num_tokens</span></td>
      </tr>
      <tr>
        <td id="L959" class="blob-num js-line-number" data-line-number="959"></td>
        <td id="LC959" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-s1>batch</span> <span class=pl-c1>in</span> <span class=pl-s1>data_iter</span>:</td>
      </tr>
      <tr>
        <td id="L960" class="blob-num js-line-number" data-line-number="960"></td>
        <td id="LC960" class="blob-code blob-code-inner js-file-line">            <span class=pl-v>X</span>, <span class=pl-v>X_vlen</span>, <span class=pl-v>Y</span>, <span class=pl-v>Y_vlen</span> <span class=pl-c1>=</span> [<span class=pl-s1>x</span>.<span class=pl-en>to</span>(<span class=pl-s1>device</span>) <span class=pl-k>for</span> <span class=pl-s1>x</span> <span class=pl-c1>in</span> <span class=pl-s1>batch</span>]</td>
      </tr>
      <tr>
        <td id="L961" class="blob-num js-line-number" data-line-number="961"></td>
        <td id="LC961" class="blob-code blob-code-inner js-file-line">            <span class=pl-v>Y_input</span>, <span class=pl-v>Y_label</span>, <span class=pl-v>Y_vlen</span> <span class=pl-c1>=</span> <span class=pl-v>Y</span>[:, :<span class=pl-c1>-</span><span class=pl-c1>1</span>], <span class=pl-v>Y</span>[:, <span class=pl-c1>1</span>:], <span class=pl-v>Y_vlen</span><span class=pl-c1>-</span><span class=pl-c1>1</span></td>
      </tr>
      <tr>
        <td id="L962" class="blob-num js-line-number" data-line-number="962"></td>
        <td id="LC962" class="blob-code blob-code-inner js-file-line">            <span class=pl-v>Y_hat</span>, <span class=pl-s1>_</span> <span class=pl-c1>=</span> <span class=pl-en>model</span>(<span class=pl-v>X</span>, <span class=pl-v>Y_input</span>, <span class=pl-v>X_vlen</span>, <span class=pl-v>Y_vlen</span>)</td>
      </tr>
      <tr>
        <td id="L963" class="blob-num js-line-number" data-line-number="963"></td>
        <td id="LC963" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>l</span> <span class=pl-c1>=</span> <span class=pl-en>loss</span>(<span class=pl-v>Y_hat</span>, <span class=pl-v>Y_label</span>, <span class=pl-v>Y_vlen</span>)</td>
      </tr>
      <tr>
        <td id="L964" class="blob-num js-line-number" data-line-number="964"></td>
        <td id="LC964" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>l</span>.<span class=pl-en>sum</span>().<span class=pl-en>backward</span>() <span class=pl-c># Making the loss scalar for backward()</span></td>
      </tr>
      <tr>
        <td id="L965" class="blob-num js-line-number" data-line-number="965"></td>
        <td id="LC965" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>d2l</span>.<span class=pl-en>grad_clipping</span>(<span class=pl-s1>model</span>, <span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L966" class="blob-num js-line-number" data-line-number="966"></td>
        <td id="LC966" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>num_tokens</span> <span class=pl-c1>=</span> <span class=pl-v>Y_vlen</span>.<span class=pl-en>sum</span>()</td>
      </tr>
      <tr>
        <td id="L967" class="blob-num js-line-number" data-line-number="967"></td>
        <td id="LC967" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>optimizer</span>.<span class=pl-en>step</span>()</td>
      </tr>
      <tr>
        <td id="L968" class="blob-num js-line-number" data-line-number="968"></td>
        <td id="LC968" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>with</span> <span class=pl-s1>torch</span>.<span class=pl-en>no_grad</span>():</td>
      </tr>
      <tr>
        <td id="L969" class="blob-num js-line-number" data-line-number="969"></td>
        <td id="LC969" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>metric</span>.<span class=pl-en>add</span>(<span class=pl-s1>l</span>.<span class=pl-en>sum</span>(), <span class=pl-s1>num_tokens</span>)</td>
      </tr>
      <tr>
        <td id="L970" class="blob-num js-line-number" data-line-number="970"></td>
        <td id="LC970" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>epoch</span> <span class=pl-c1>%</span> <span class=pl-c1>10</span> <span class=pl-c1>==</span> <span class=pl-c1>0</span>:</td>
      </tr>
      <tr>
        <td id="L971" class="blob-num js-line-number" data-line-number="971"></td>
        <td id="LC971" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>animator</span>.<span class=pl-en>add</span>(<span class=pl-s1>epoch</span>, (<span class=pl-s1>metric</span>[<span class=pl-c1>0</span>]<span class=pl-c1>/</span><span class=pl-s1>metric</span>[<span class=pl-c1>1</span>],))</td>
      </tr>
      <tr>
        <td id="L972" class="blob-num js-line-number" data-line-number="972"></td>
        <td id="LC972" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>f&#39;loss <span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>metric</span>[<span class=pl-c1>0</span>] <span class=pl-c1>/</span> <span class=pl-s1>metric</span>[<span class=pl-c1>1</span>]:.3f<span class=pl-kos>}</span></span>, <span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>metric</span>[<span class=pl-c1>1</span>] <span class=pl-c1>/</span> <span class=pl-s1>timer</span>.<span class=pl-en>stop</span>():.1f<span class=pl-kos>}</span></span> &#39;</span></td>
      </tr>
      <tr>
        <td id="L973" class="blob-num js-line-number" data-line-number="973"></td>
        <td id="LC973" class="blob-code blob-code-inner js-file-line">          <span class=pl-s>f&#39;tokens/sec on <span class=pl-s1><span class=pl-kos>{</span><span class=pl-en>str</span>(<span class=pl-s1>device</span>)<span class=pl-kos>}</span></span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L974" class="blob-num js-line-number" data-line-number="974"></td>
        <td id="LC974" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L975" class="blob-num js-line-number" data-line-number="975"></td>
        <td id="LC975" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L976" class="blob-num js-line-number" data-line-number="976"></td>
        <td id="LC976" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_recurrent-modern/seq2seq.md</span></td>
      </tr>
      <tr>
        <td id="L977" class="blob-num js-line-number" data-line-number="977"></td>
        <td id="LC977" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>predict_s2s_ch9</span>(<span class=pl-s1>model</span>, <span class=pl-s1>src_sentence</span>, <span class=pl-s1>src_vocab</span>, <span class=pl-s1>tgt_vocab</span>, <span class=pl-s1>num_steps</span>,</td>
      </tr>
      <tr>
        <td id="L978" class="blob-num js-line-number" data-line-number="978"></td>
        <td id="LC978" class="blob-code blob-code-inner js-file-line">                    <span class=pl-s1>device</span>):</td>
      </tr>
      <tr>
        <td id="L979" class="blob-num js-line-number" data-line-number="979"></td>
        <td id="LC979" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>src_tokens</span> <span class=pl-c1>=</span> <span class=pl-s1>src_vocab</span>[<span class=pl-s1>src_sentence</span>.<span class=pl-en>lower</span>().<span class=pl-en>split</span>(<span class=pl-s>&#39; &#39;</span>)]</td>
      </tr>
      <tr>
        <td id="L980" class="blob-num js-line-number" data-line-number="980"></td>
        <td id="LC980" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>enc_valid_len</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-en>tensor</span>([<span class=pl-en>len</span>(<span class=pl-s1>src_tokens</span>)], <span class=pl-s1>device</span><span class=pl-c1>=</span><span class=pl-s1>device</span>)</td>
      </tr>
      <tr>
        <td id="L981" class="blob-num js-line-number" data-line-number="981"></td>
        <td id="LC981" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>src_tokens</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>truncate_pad</span>(<span class=pl-s1>src_tokens</span>, <span class=pl-s1>num_steps</span>, <span class=pl-s1>src_vocab</span>[<span class=pl-s>&#39;&lt;pad&gt;&#39;</span>])</td>
      </tr>
      <tr>
        <td id="L982" class="blob-num js-line-number" data-line-number="982"></td>
        <td id="LC982" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>enc_X</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-en>tensor</span>(<span class=pl-s1>src_tokens</span>, <span class=pl-s1>dtype</span><span class=pl-c1>=</span><span class=pl-s1>torch</span>.<span class=pl-s1>long</span>, <span class=pl-s1>device</span><span class=pl-c1>=</span><span class=pl-s1>device</span>)</td>
      </tr>
      <tr>
        <td id="L983" class="blob-num js-line-number" data-line-number="983"></td>
        <td id="LC983" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Add the  batch size dimension</span></td>
      </tr>
      <tr>
        <td id="L984" class="blob-num js-line-number" data-line-number="984"></td>
        <td id="LC984" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>enc_outputs</span> <span class=pl-c1>=</span> <span class=pl-s1>model</span>.<span class=pl-en>encoder</span>(<span class=pl-s1>torch</span>.<span class=pl-en>unsqueeze</span>(<span class=pl-s1>enc_X</span>, <span class=pl-s1>dim</span><span class=pl-c1>=</span><span class=pl-c1>0</span>),</td>
      </tr>
      <tr>
        <td id="L985" class="blob-num js-line-number" data-line-number="985"></td>
        <td id="LC985" class="blob-code blob-code-inner js-file-line">                                <span class=pl-s1>enc_valid_len</span>)</td>
      </tr>
      <tr>
        <td id="L986" class="blob-num js-line-number" data-line-number="986"></td>
        <td id="LC986" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>dec_state</span> <span class=pl-c1>=</span> <span class=pl-s1>model</span>.<span class=pl-s1>decoder</span>.<span class=pl-en>init_state</span>(<span class=pl-s1>enc_outputs</span>, <span class=pl-s1>enc_valid_len</span>)</td>
      </tr>
      <tr>
        <td id="L987" class="blob-num js-line-number" data-line-number="987"></td>
        <td id="LC987" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>dec_X</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-en>unsqueeze</span>(<span class=pl-s1>torch</span>.<span class=pl-en>tensor</span>([<span class=pl-s1>tgt_vocab</span>[<span class=pl-s>&#39;&lt;bos&gt;&#39;</span>]], <span class=pl-s1>dtype</span><span class=pl-c1>=</span><span class=pl-s1>torch</span>.<span class=pl-s1>long</span>, <span class=pl-s1>device</span><span class=pl-c1>=</span><span class=pl-s1>device</span>), <span class=pl-s1>dim</span><span class=pl-c1>=</span><span class=pl-c1>0</span>)</td>
      </tr>
      <tr>
        <td id="L988" class="blob-num js-line-number" data-line-number="988"></td>
        <td id="LC988" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>predict_tokens</span> <span class=pl-c1>=</span> []</td>
      </tr>
      <tr>
        <td id="L989" class="blob-num js-line-number" data-line-number="989"></td>
        <td id="LC989" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>_</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-s1>num_steps</span>):</td>
      </tr>
      <tr>
        <td id="L990" class="blob-num js-line-number" data-line-number="990"></td>
        <td id="LC990" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>Y</span>, <span class=pl-s1>dec_state</span> <span class=pl-c1>=</span> <span class=pl-s1>model</span>.<span class=pl-en>decoder</span>(<span class=pl-s1>dec_X</span>, <span class=pl-s1>dec_state</span>)</td>
      </tr>
      <tr>
        <td id="L991" class="blob-num js-line-number" data-line-number="991"></td>
        <td id="LC991" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># The token with highest score is used as the next timestep input</span></td>
      </tr>
      <tr>
        <td id="L992" class="blob-num js-line-number" data-line-number="992"></td>
        <td id="LC992" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>dec_X</span> <span class=pl-c1>=</span> <span class=pl-v>Y</span>.<span class=pl-en>argmax</span>(<span class=pl-s1>dim</span><span class=pl-c1>=</span><span class=pl-c1>2</span>)</td>
      </tr>
      <tr>
        <td id="L993" class="blob-num js-line-number" data-line-number="993"></td>
        <td id="LC993" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>py</span> <span class=pl-c1>=</span> <span class=pl-s1>dec_X</span>.<span class=pl-en>squeeze</span>(<span class=pl-s1>dim</span><span class=pl-c1>=</span><span class=pl-c1>0</span>).<span class=pl-en>type</span>(<span class=pl-s1>torch</span>.<span class=pl-s1>int32</span>).<span class=pl-en>item</span>()</td>
      </tr>
      <tr>
        <td id="L994" class="blob-num js-line-number" data-line-number="994"></td>
        <td id="LC994" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>py</span> <span class=pl-c1>==</span> <span class=pl-s1>tgt_vocab</span>[<span class=pl-s>&#39;&lt;eos&gt;&#39;</span>]:</td>
      </tr>
      <tr>
        <td id="L995" class="blob-num js-line-number" data-line-number="995"></td>
        <td id="LC995" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>break</span></td>
      </tr>
      <tr>
        <td id="L996" class="blob-num js-line-number" data-line-number="996"></td>
        <td id="LC996" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>predict_tokens</span>.<span class=pl-en>append</span>(<span class=pl-s1>py</span>)</td>
      </tr>
      <tr>
        <td id="L997" class="blob-num js-line-number" data-line-number="997"></td>
        <td id="LC997" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s>&#39; &#39;</span>.<span class=pl-en>join</span>(<span class=pl-s1>tgt_vocab</span>.<span class=pl-en>to_tokens</span>(<span class=pl-s1>predict_tokens</span>))</td>
      </tr>
      <tr>
        <td id="L998" class="blob-num js-line-number" data-line-number="998"></td>
        <td id="LC998" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L999" class="blob-num js-line-number" data-line-number="999"></td>
        <td id="LC999" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1000" class="blob-num js-line-number" data-line-number="1000"></td>
        <td id="LC1000" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_attention-mechanisms/attention.md</span></td>
      </tr>
      <tr>
        <td id="L1001" class="blob-num js-line-number" data-line-number="1001"></td>
        <td id="LC1001" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>masked_softmax</span>(<span class=pl-v>X</span>, <span class=pl-s1>valid_len</span>):</td>
      </tr>
      <tr>
        <td id="L1002" class="blob-num js-line-number" data-line-number="1002"></td>
        <td id="LC1002" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Perform softmax by filtering out some elements.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L1003" class="blob-num js-line-number" data-line-number="1003"></td>
        <td id="LC1003" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># X: 3-D tensor, valid_len: 1-D or 2-D tensor</span></td>
      </tr>
      <tr>
        <td id="L1004" class="blob-num js-line-number" data-line-number="1004"></td>
        <td id="LC1004" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>if</span> <span class=pl-s1>valid_len</span> <span class=pl-c1>is</span> <span class=pl-c1>None</span>:</td>
      </tr>
      <tr>
        <td id="L1005" class="blob-num js-line-number" data-line-number="1005"></td>
        <td id="LC1005" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>nn</span>.<span class=pl-s1>functional</span>.<span class=pl-en>softmax</span>(<span class=pl-v>X</span>, <span class=pl-s1>dim</span><span class=pl-c1>=</span><span class=pl-c1>-</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L1006" class="blob-num js-line-number" data-line-number="1006"></td>
        <td id="LC1006" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>else</span>:</td>
      </tr>
      <tr>
        <td id="L1007" class="blob-num js-line-number" data-line-number="1007"></td>
        <td id="LC1007" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>shape</span> <span class=pl-c1>=</span> <span class=pl-v>X</span>.<span class=pl-s1>shape</span></td>
      </tr>
      <tr>
        <td id="L1008" class="blob-num js-line-number" data-line-number="1008"></td>
        <td id="LC1008" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-s1>valid_len</span>.<span class=pl-en>dim</span>() <span class=pl-c1>==</span> <span class=pl-c1>1</span>:</td>
      </tr>
      <tr>
        <td id="L1009" class="blob-num js-line-number" data-line-number="1009"></td>
        <td id="LC1009" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>valid_len</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-en>repeat_interleave</span>(<span class=pl-s1>valid_len</span>, <span class=pl-s1>repeats</span><span class=pl-c1>=</span><span class=pl-s1>shape</span>[<span class=pl-c1>1</span>],</td>
      </tr>
      <tr>
        <td id="L1010" class="blob-num js-line-number" data-line-number="1010"></td>
        <td id="LC1010" class="blob-code blob-code-inner js-file-line">                                                <span class=pl-s1>dim</span><span class=pl-c1>=</span><span class=pl-c1>0</span>)</td>
      </tr>
      <tr>
        <td id="L1011" class="blob-num js-line-number" data-line-number="1011"></td>
        <td id="LC1011" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>else</span>:</td>
      </tr>
      <tr>
        <td id="L1012" class="blob-num js-line-number" data-line-number="1012"></td>
        <td id="LC1012" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>valid_len</span> <span class=pl-c1>=</span> <span class=pl-s1>valid_len</span>.<span class=pl-en>reshape</span>(<span class=pl-c1>-</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L1013" class="blob-num js-line-number" data-line-number="1013"></td>
        <td id="LC1013" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># Fill masked elements with a large negative, whose exp is 0</span></td>
      </tr>
      <tr>
        <td id="L1014" class="blob-num js-line-number" data-line-number="1014"></td>
        <td id="LC1014" class="blob-code blob-code-inner js-file-line">        <span class=pl-v>X</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>sequence_mask</span>(<span class=pl-v>X</span>.<span class=pl-en>reshape</span>(<span class=pl-c1>-</span><span class=pl-c1>1</span>, <span class=pl-s1>shape</span>[<span class=pl-c1>-</span><span class=pl-c1>1</span>]), <span class=pl-s1>valid_len</span>, <span class=pl-s1>value</span><span class=pl-c1>=</span><span class=pl-c1>-</span><span class=pl-c1>1e6</span>)</td>
      </tr>
      <tr>
        <td id="L1015" class="blob-num js-line-number" data-line-number="1015"></td>
        <td id="LC1015" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>nn</span>.<span class=pl-s1>functional</span>.<span class=pl-en>softmax</span>(<span class=pl-v>X</span>.<span class=pl-en>reshape</span>(<span class=pl-s1>shape</span>), <span class=pl-s1>dim</span><span class=pl-c1>=</span><span class=pl-c1>-</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L1016" class="blob-num js-line-number" data-line-number="1016"></td>
        <td id="LC1016" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1017" class="blob-num js-line-number" data-line-number="1017"></td>
        <td id="LC1017" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1018" class="blob-num js-line-number" data-line-number="1018"></td>
        <td id="LC1018" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_attention-mechanisms/attention.md</span></td>
      </tr>
      <tr>
        <td id="L1019" class="blob-num js-line-number" data-line-number="1019"></td>
        <td id="LC1019" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>DotProductAttention</span>(<span class=pl-s1>nn</span>.<span class=pl-v>Module</span>):</td>
      </tr>
      <tr>
        <td id="L1020" class="blob-num js-line-number" data-line-number="1020"></td>
        <td id="LC1020" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>, <span class=pl-s1>dropout</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>):</td>
      </tr>
      <tr>
        <td id="L1021" class="blob-num js-line-number" data-line-number="1021"></td>
        <td id="LC1021" class="blob-code blob-code-inner js-file-line">        <span class=pl-en>super</span>(<span class=pl-v>DotProductAttention</span>, <span class=pl-s1>self</span>).<span class=pl-en>__init__</span>(<span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L1022" class="blob-num js-line-number" data-line-number="1022"></td>
        <td id="LC1022" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>dropout</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>Dropout</span>(<span class=pl-s1>dropout</span>)</td>
      </tr>
      <tr>
        <td id="L1023" class="blob-num js-line-number" data-line-number="1023"></td>
        <td id="LC1023" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1024" class="blob-num js-line-number" data-line-number="1024"></td>
        <td id="LC1024" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># `query`: (`batch_size`, #queries, `d`)</span></td>
      </tr>
      <tr>
        <td id="L1025" class="blob-num js-line-number" data-line-number="1025"></td>
        <td id="LC1025" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># `key`: (`batch_size`, #kv_pairs, `d`)</span></td>
      </tr>
      <tr>
        <td id="L1026" class="blob-num js-line-number" data-line-number="1026"></td>
        <td id="LC1026" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># `value`: (`batch_size`, #kv_pairs, `dim_v`)</span></td>
      </tr>
      <tr>
        <td id="L1027" class="blob-num js-line-number" data-line-number="1027"></td>
        <td id="LC1027" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># `valid_len`: either (`batch_size`, ) or (`batch_size`, xx)</span></td>
      </tr>
      <tr>
        <td id="L1028" class="blob-num js-line-number" data-line-number="1028"></td>
        <td id="LC1028" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>forward</span>(<span class=pl-s1>self</span>, <span class=pl-s1>query</span>, <span class=pl-s1>key</span>, <span class=pl-s1>value</span>, <span class=pl-s1>valid_len</span><span class=pl-c1>=</span><span class=pl-c1>None</span>):</td>
      </tr>
      <tr>
        <td id="L1029" class="blob-num js-line-number" data-line-number="1029"></td>
        <td id="LC1029" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>d</span> <span class=pl-c1>=</span> <span class=pl-s1>query</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>-</span><span class=pl-c1>1</span>]</td>
      </tr>
      <tr>
        <td id="L1030" class="blob-num js-line-number" data-line-number="1030"></td>
        <td id="LC1030" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># Set transpose_b=True to swap the last two dimensions of key</span></td>
      </tr>
      <tr>
        <td id="L1031" class="blob-num js-line-number" data-line-number="1031"></td>
        <td id="LC1031" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>scores</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-en>bmm</span>(<span class=pl-s1>query</span>, <span class=pl-s1>key</span>.<span class=pl-en>transpose</span>(<span class=pl-c1>1</span>,<span class=pl-c1>2</span>)) <span class=pl-c1>/</span> <span class=pl-s1>math</span>.<span class=pl-en>sqrt</span>(<span class=pl-s1>d</span>)</td>
      </tr>
      <tr>
        <td id="L1032" class="blob-num js-line-number" data-line-number="1032"></td>
        <td id="LC1032" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>attention_weights</span> <span class=pl-c1>=</span> <span class=pl-s1>self</span>.<span class=pl-en>dropout</span>(<span class=pl-en>masked_softmax</span>(<span class=pl-s1>scores</span>, <span class=pl-s1>valid_len</span>))</td>
      </tr>
      <tr>
        <td id="L1033" class="blob-num js-line-number" data-line-number="1033"></td>
        <td id="LC1033" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>torch</span>.<span class=pl-en>bmm</span>(<span class=pl-s1>attention_weights</span>, <span class=pl-s1>value</span>)</td>
      </tr>
      <tr>
        <td id="L1034" class="blob-num js-line-number" data-line-number="1034"></td>
        <td id="LC1034" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1035" class="blob-num js-line-number" data-line-number="1035"></td>
        <td id="LC1035" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1036" class="blob-num js-line-number" data-line-number="1036"></td>
        <td id="LC1036" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_attention-mechanisms/attention.md</span></td>
      </tr>
      <tr>
        <td id="L1037" class="blob-num js-line-number" data-line-number="1037"></td>
        <td id="LC1037" class="blob-code blob-code-inner js-file-line"><span class=pl-k>class</span> <span class=pl-v>MLPAttention</span>(<span class=pl-s1>nn</span>.<span class=pl-v>Module</span>):</td>
      </tr>
      <tr>
        <td id="L1038" class="blob-num js-line-number" data-line-number="1038"></td>
        <td id="LC1038" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>__init__</span>(<span class=pl-s1>self</span>, <span class=pl-s1>key_size</span>, <span class=pl-s1>query_size</span>, <span class=pl-s1>units</span>, <span class=pl-s1>dropout</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>):</td>
      </tr>
      <tr>
        <td id="L1039" class="blob-num js-line-number" data-line-number="1039"></td>
        <td id="LC1039" class="blob-code blob-code-inner js-file-line">        <span class=pl-en>super</span>(<span class=pl-v>MLPAttention</span>, <span class=pl-s1>self</span>).<span class=pl-en>__init__</span>(<span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L1040" class="blob-num js-line-number" data-line-number="1040"></td>
        <td id="LC1040" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-v>W_k</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>Linear</span>(<span class=pl-s1>key_size</span>, <span class=pl-s1>units</span>, <span class=pl-s1>bias</span><span class=pl-c1>=</span><span class=pl-c1>False</span>)</td>
      </tr>
      <tr>
        <td id="L1041" class="blob-num js-line-number" data-line-number="1041"></td>
        <td id="LC1041" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-v>W_q</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>Linear</span>(<span class=pl-s1>query_size</span>, <span class=pl-s1>units</span>, <span class=pl-s1>bias</span><span class=pl-c1>=</span><span class=pl-c1>False</span>)</td>
      </tr>
      <tr>
        <td id="L1042" class="blob-num js-line-number" data-line-number="1042"></td>
        <td id="LC1042" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>v</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>Linear</span>(<span class=pl-s1>units</span>, <span class=pl-c1>1</span>, <span class=pl-s1>bias</span><span class=pl-c1>=</span><span class=pl-c1>False</span>)</td>
      </tr>
      <tr>
        <td id="L1043" class="blob-num js-line-number" data-line-number="1043"></td>
        <td id="LC1043" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>self</span>.<span class=pl-s1>dropout</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>Dropout</span>(<span class=pl-s1>dropout</span>)</td>
      </tr>
      <tr>
        <td id="L1044" class="blob-num js-line-number" data-line-number="1044"></td>
        <td id="LC1044" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1045" class="blob-num js-line-number" data-line-number="1045"></td>
        <td id="LC1045" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>forward</span>(<span class=pl-s1>self</span>, <span class=pl-s1>query</span>, <span class=pl-s1>key</span>, <span class=pl-s1>value</span>, <span class=pl-s1>valid_len</span>):</td>
      </tr>
      <tr>
        <td id="L1046" class="blob-num js-line-number" data-line-number="1046"></td>
        <td id="LC1046" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>query</span>, <span class=pl-s1>key</span> <span class=pl-c1>=</span> <span class=pl-s1>self</span>.<span class=pl-v>W_q</span>(<span class=pl-s1>query</span>), <span class=pl-s1>self</span>.<span class=pl-v>W_k</span>(<span class=pl-s1>key</span>)</td>
      </tr>
      <tr>
        <td id="L1047" class="blob-num js-line-number" data-line-number="1047"></td>
        <td id="LC1047" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># Expand query to (`batch_size`, #queries, 1, units), and key to</span></td>
      </tr>
      <tr>
        <td id="L1048" class="blob-num js-line-number" data-line-number="1048"></td>
        <td id="LC1048" class="blob-code blob-code-inner js-file-line">        <span class=pl-c># (`batch_size`, 1, #kv_pairs, units). Then plus them with broadcast</span></td>
      </tr>
      <tr>
        <td id="L1049" class="blob-num js-line-number" data-line-number="1049"></td>
        <td id="LC1049" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>features</span> <span class=pl-c1>=</span> <span class=pl-s1>query</span>.<span class=pl-en>unsqueeze</span>(<span class=pl-c1>2</span>) <span class=pl-c1>+</span> <span class=pl-s1>key</span>.<span class=pl-en>unsqueeze</span>(<span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L1050" class="blob-num js-line-number" data-line-number="1050"></td>
        <td id="LC1050" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>features</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-en>tanh</span>(<span class=pl-s1>features</span>)</td>
      </tr>
      <tr>
        <td id="L1051" class="blob-num js-line-number" data-line-number="1051"></td>
        <td id="LC1051" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>scores</span> <span class=pl-c1>=</span> <span class=pl-s1>self</span>.<span class=pl-en>v</span>(<span class=pl-s1>features</span>).<span class=pl-en>squeeze</span>(<span class=pl-c1>-</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L1052" class="blob-num js-line-number" data-line-number="1052"></td>
        <td id="LC1052" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>attention_weights</span> <span class=pl-c1>=</span> <span class=pl-s1>self</span>.<span class=pl-en>dropout</span>(<span class=pl-en>masked_softmax</span>(<span class=pl-s1>scores</span>, <span class=pl-s1>valid_len</span>))</td>
      </tr>
      <tr>
        <td id="L1053" class="blob-num js-line-number" data-line-number="1053"></td>
        <td id="LC1053" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>return</span> <span class=pl-s1>torch</span>.<span class=pl-en>bmm</span>(<span class=pl-s1>attention_weights</span>, <span class=pl-s1>value</span>)</td>
      </tr>
      <tr>
        <td id="L1054" class="blob-num js-line-number" data-line-number="1054"></td>
        <td id="LC1054" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1055" class="blob-num js-line-number" data-line-number="1055"></td>
        <td id="LC1055" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1056" class="blob-num js-line-number" data-line-number="1056"></td>
        <td id="LC1056" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_optimization/optimization-intro.md</span></td>
      </tr>
      <tr>
        <td id="L1057" class="blob-num js-line-number" data-line-number="1057"></td>
        <td id="LC1057" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>annotate</span>(<span class=pl-s1>text</span>, <span class=pl-s1>xy</span>, <span class=pl-s1>xytext</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L1058" class="blob-num js-line-number" data-line-number="1058"></td>
        <td id="LC1058" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>d2l</span>.<span class=pl-s1>plt</span>.<span class=pl-en>gca</span>().<span class=pl-en>annotate</span>(<span class=pl-s1>text</span>, <span class=pl-s1>xy</span><span class=pl-c1>=</span><span class=pl-s1>xy</span>, <span class=pl-s1>xytext</span><span class=pl-c1>=</span><span class=pl-s1>xytext</span>,</td>
      </tr>
      <tr>
        <td id="L1059" class="blob-num js-line-number" data-line-number="1059"></td>
        <td id="LC1059" class="blob-code blob-code-inner js-file-line">                           <span class=pl-s1>arrowprops</span><span class=pl-c1>=</span><span class=pl-en>dict</span>(<span class=pl-s1>arrowstyle</span><span class=pl-c1>=</span><span class=pl-s>&#39;-&gt;&#39;</span>))</td>
      </tr>
      <tr>
        <td id="L1060" class="blob-num js-line-number" data-line-number="1060"></td>
        <td id="LC1060" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1061" class="blob-num js-line-number" data-line-number="1061"></td>
        <td id="LC1061" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1062" class="blob-num js-line-number" data-line-number="1062"></td>
        <td id="LC1062" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_optimization/gd.md</span></td>
      </tr>
      <tr>
        <td id="L1063" class="blob-num js-line-number" data-line-number="1063"></td>
        <td id="LC1063" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>train_2d</span>(<span class=pl-s1>trainer</span>, <span class=pl-s1>steps</span><span class=pl-c1>=</span><span class=pl-c1>20</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L1064" class="blob-num js-line-number" data-line-number="1064"></td>
        <td id="LC1064" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Optimize a 2-dim objective function with a customized trainer.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L1065" class="blob-num js-line-number" data-line-number="1065"></td>
        <td id="LC1065" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># s1 and s2 are internal state variables and will</span></td>
      </tr>
      <tr>
        <td id="L1066" class="blob-num js-line-number" data-line-number="1066"></td>
        <td id="LC1066" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># be used later in the chapter</span></td>
      </tr>
      <tr>
        <td id="L1067" class="blob-num js-line-number" data-line-number="1067"></td>
        <td id="LC1067" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>x1</span>, <span class=pl-s1>x2</span>, <span class=pl-s1>s1</span>, <span class=pl-s1>s2</span> <span class=pl-c1>=</span> <span class=pl-c1>-</span><span class=pl-c1>5</span>, <span class=pl-c1>-</span><span class=pl-c1>2</span>, <span class=pl-c1>0</span>, <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L1068" class="blob-num js-line-number" data-line-number="1068"></td>
        <td id="LC1068" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>results</span> <span class=pl-c1>=</span> [(<span class=pl-s1>x1</span>, <span class=pl-s1>x2</span>)]</td>
      </tr>
      <tr>
        <td id="L1069" class="blob-num js-line-number" data-line-number="1069"></td>
        <td id="LC1069" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>i</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-s1>steps</span>):</td>
      </tr>
      <tr>
        <td id="L1070" class="blob-num js-line-number" data-line-number="1070"></td>
        <td id="LC1070" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>x1</span>, <span class=pl-s1>x2</span>, <span class=pl-s1>s1</span>, <span class=pl-s1>s2</span> <span class=pl-c1>=</span> <span class=pl-en>trainer</span>(<span class=pl-s1>x1</span>, <span class=pl-s1>x2</span>, <span class=pl-s1>s1</span>, <span class=pl-s1>s2</span>)</td>
      </tr>
      <tr>
        <td id="L1071" class="blob-num js-line-number" data-line-number="1071"></td>
        <td id="LC1071" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>results</span>.<span class=pl-en>append</span>((<span class=pl-s1>x1</span>, <span class=pl-s1>x2</span>))</td>
      </tr>
      <tr>
        <td id="L1072" class="blob-num js-line-number" data-line-number="1072"></td>
        <td id="LC1072" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>results</span></td>
      </tr>
      <tr>
        <td id="L1073" class="blob-num js-line-number" data-line-number="1073"></td>
        <td id="LC1073" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1074" class="blob-num js-line-number" data-line-number="1074"></td>
        <td id="LC1074" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1075" class="blob-num js-line-number" data-line-number="1075"></td>
        <td id="LC1075" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_optimization/gd.md</span></td>
      </tr>
      <tr>
        <td id="L1076" class="blob-num js-line-number" data-line-number="1076"></td>
        <td id="LC1076" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>show_trace_2d</span>(<span class=pl-s1>f</span>, <span class=pl-s1>results</span>):  <span class=pl-c>#@save</span></td>
      </tr>
      <tr>
        <td id="L1077" class="blob-num js-line-number" data-line-number="1077"></td>
        <td id="LC1077" class="blob-code blob-code-inner js-file-line">    <span class=pl-s>&quot;&quot;&quot;Show the trace of 2D variables during optimization.&quot;&quot;&quot;</span></td>
      </tr>
      <tr>
        <td id="L1078" class="blob-num js-line-number" data-line-number="1078"></td>
        <td id="LC1078" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>d2l</span>.<span class=pl-en>set_figsize</span>()</td>
      </tr>
      <tr>
        <td id="L1079" class="blob-num js-line-number" data-line-number="1079"></td>
        <td id="LC1079" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>d2l</span>.<span class=pl-s1>plt</span>.<span class=pl-en>plot</span>(<span class=pl-c1>*</span><span class=pl-en>zip</span>(<span class=pl-c1>*</span><span class=pl-s1>results</span>), <span class=pl-s>&#39;-o&#39;</span>, <span class=pl-s1>color</span><span class=pl-c1>=</span><span class=pl-s>&#39;#ff7f0e&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L1080" class="blob-num js-line-number" data-line-number="1080"></td>
        <td id="LC1080" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>x1</span>, <span class=pl-s1>x2</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>meshgrid</span>(<span class=pl-s1>d2l</span>.<span class=pl-en>arange</span>(<span class=pl-c1>-</span><span class=pl-c1>5.5</span>, <span class=pl-c1>1.0</span>, <span class=pl-c1>0.1</span>),</td>
      </tr>
      <tr>
        <td id="L1081" class="blob-num js-line-number" data-line-number="1081"></td>
        <td id="LC1081" class="blob-code blob-code-inner js-file-line">                          <span class=pl-s1>d2l</span>.<span class=pl-en>arange</span>(<span class=pl-c1>-</span><span class=pl-c1>3.0</span>, <span class=pl-c1>1.0</span>, <span class=pl-c1>0.1</span>))</td>
      </tr>
      <tr>
        <td id="L1082" class="blob-num js-line-number" data-line-number="1082"></td>
        <td id="LC1082" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>d2l</span>.<span class=pl-s1>plt</span>.<span class=pl-en>contour</span>(<span class=pl-s1>x1</span>, <span class=pl-s1>x2</span>, <span class=pl-en>f</span>(<span class=pl-s1>x1</span>, <span class=pl-s1>x2</span>), <span class=pl-s1>colors</span><span class=pl-c1>=</span><span class=pl-s>&#39;#1f77b4&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L1083" class="blob-num js-line-number" data-line-number="1083"></td>
        <td id="LC1083" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>d2l</span>.<span class=pl-s1>plt</span>.<span class=pl-en>xlabel</span>(<span class=pl-s>&#39;x1&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L1084" class="blob-num js-line-number" data-line-number="1084"></td>
        <td id="LC1084" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>d2l</span>.<span class=pl-s1>plt</span>.<span class=pl-en>ylabel</span>(<span class=pl-s>&#39;x2&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L1085" class="blob-num js-line-number" data-line-number="1085"></td>
        <td id="LC1085" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1086" class="blob-num js-line-number" data-line-number="1086"></td>
        <td id="LC1086" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1087" class="blob-num js-line-number" data-line-number="1087"></td>
        <td id="LC1087" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_optimization/minibatch-sgd.md</span></td>
      </tr>
      <tr>
        <td id="L1088" class="blob-num js-line-number" data-line-number="1088"></td>
        <td id="LC1088" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>d2l</span>.<span class=pl-v>DATA_HUB</span>[<span class=pl-s>&#39;airfoil&#39;</span>] <span class=pl-c1>=</span> (<span class=pl-s1>d2l</span>.<span class=pl-v>DATA_URL</span> <span class=pl-c1>+</span> <span class=pl-s>&#39;airfoil_self_noise.dat&#39;</span>,</td>
      </tr>
      <tr>
        <td id="L1089" class="blob-num js-line-number" data-line-number="1089"></td>
        <td id="LC1089" class="blob-code blob-code-inner js-file-line">                           <span class=pl-s>&#39;76e5be1548fd8222e5074cf0faae75edff8cf93f&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L1090" class="blob-num js-line-number" data-line-number="1090"></td>
        <td id="LC1090" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1091" class="blob-num js-line-number" data-line-number="1091"></td>
        <td id="LC1091" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1092" class="blob-num js-line-number" data-line-number="1092"></td>
        <td id="LC1092" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_optimization/minibatch-sgd.md</span></td>
      </tr>
      <tr>
        <td id="L1093" class="blob-num js-line-number" data-line-number="1093"></td>
        <td id="LC1093" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>get_data_ch11</span>(<span class=pl-s1>batch_size</span><span class=pl-c1>=</span><span class=pl-c1>10</span>, <span class=pl-s1>n</span><span class=pl-c1>=</span><span class=pl-c1>1500</span>):</td>
      </tr>
      <tr>
        <td id="L1094" class="blob-num js-line-number" data-line-number="1094"></td>
        <td id="LC1094" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>data</span> <span class=pl-c1>=</span> <span class=pl-s1>np</span>.<span class=pl-en>genfromtxt</span>(<span class=pl-s1>d2l</span>.<span class=pl-en>download</span>(<span class=pl-s>&#39;airfoil&#39;</span>),</td>
      </tr>
      <tr>
        <td id="L1095" class="blob-num js-line-number" data-line-number="1095"></td>
        <td id="LC1095" class="blob-code blob-code-inner js-file-line">                         <span class=pl-s1>dtype</span><span class=pl-c1>=</span><span class=pl-s1>np</span>.<span class=pl-s1>float32</span>, <span class=pl-s1>delimiter</span><span class=pl-c1>=</span><span class=pl-s>&#39;<span class=pl-cce>\t</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L1096" class="blob-num js-line-number" data-line-number="1096"></td>
        <td id="LC1096" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>data</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-en>from_numpy</span>((<span class=pl-s1>data</span> <span class=pl-c1>-</span> <span class=pl-s1>data</span>.<span class=pl-en>mean</span>(<span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>0</span>)) <span class=pl-c1>/</span> <span class=pl-s1>data</span>.<span class=pl-en>std</span>(<span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>0</span>))</td>
      </tr>
      <tr>
        <td id="L1097" class="blob-num js-line-number" data-line-number="1097"></td>
        <td id="LC1097" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>data_iter</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-en>load_array</span>((<span class=pl-s1>data</span>[:<span class=pl-s1>n</span>, :<span class=pl-c1>-</span><span class=pl-c1>1</span>], <span class=pl-s1>data</span>[:<span class=pl-s1>n</span>, <span class=pl-c1>-</span><span class=pl-c1>1</span>]),</td>
      </tr>
      <tr>
        <td id="L1098" class="blob-num js-line-number" data-line-number="1098"></td>
        <td id="LC1098" class="blob-code blob-code-inner js-file-line">                               <span class=pl-s1>batch_size</span>, <span class=pl-s1>is_train</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L1099" class="blob-num js-line-number" data-line-number="1099"></td>
        <td id="LC1099" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>data_iter</span>, <span class=pl-s1>data</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>1</span>]<span class=pl-c1>-</span><span class=pl-c1>1</span></td>
      </tr>
      <tr>
        <td id="L1100" class="blob-num js-line-number" data-line-number="1100"></td>
        <td id="LC1100" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1101" class="blob-num js-line-number" data-line-number="1101"></td>
        <td id="LC1101" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1102" class="blob-num js-line-number" data-line-number="1102"></td>
        <td id="LC1102" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_optimization/minibatch-sgd.md</span></td>
      </tr>
      <tr>
        <td id="L1103" class="blob-num js-line-number" data-line-number="1103"></td>
        <td id="LC1103" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>train_ch11</span>(<span class=pl-s1>trainer_fn</span>, <span class=pl-s1>states</span>, <span class=pl-s1>hyperparams</span>, <span class=pl-s1>data_iter</span>,</td>
      </tr>
      <tr>
        <td id="L1104" class="blob-num js-line-number" data-line-number="1104"></td>
        <td id="LC1104" class="blob-code blob-code-inner js-file-line">               <span class=pl-s1>feature_dim</span>, <span class=pl-s1>num_epochs</span><span class=pl-c1>=</span><span class=pl-c1>2</span>):</td>
      </tr>
      <tr>
        <td id="L1105" class="blob-num js-line-number" data-line-number="1105"></td>
        <td id="LC1105" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Initialization</span></td>
      </tr>
      <tr>
        <td id="L1106" class="blob-num js-line-number" data-line-number="1106"></td>
        <td id="LC1106" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>w</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-en>normal</span>(<span class=pl-s1>mean</span><span class=pl-c1>=</span><span class=pl-c1>0.0</span>, <span class=pl-s1>std</span><span class=pl-c1>=</span><span class=pl-c1>0.01</span>, <span class=pl-s1>size</span><span class=pl-c1>=</span>(<span class=pl-s1>feature_dim</span>, <span class=pl-c1>1</span>),</td>
      </tr>
      <tr>
        <td id="L1107" class="blob-num js-line-number" data-line-number="1107"></td>
        <td id="LC1107" class="blob-code blob-code-inner js-file-line">                     <span class=pl-s1>requires_grad</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L1108" class="blob-num js-line-number" data-line-number="1108"></td>
        <td id="LC1108" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>b</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-en>zeros</span>((<span class=pl-c1>1</span>), <span class=pl-s1>requires_grad</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L1109" class="blob-num js-line-number" data-line-number="1109"></td>
        <td id="LC1109" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>net</span>, <span class=pl-s1>loss</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span> <span class=pl-v>X</span>: <span class=pl-s1>d2l</span>.<span class=pl-en>linreg</span>(<span class=pl-v>X</span>, <span class=pl-s1>w</span>, <span class=pl-s1>b</span>), <span class=pl-s1>d2l</span>.<span class=pl-s1>squared_loss</span></td>
      </tr>
      <tr>
        <td id="L1110" class="blob-num js-line-number" data-line-number="1110"></td>
        <td id="LC1110" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Train</span></td>
      </tr>
      <tr>
        <td id="L1111" class="blob-num js-line-number" data-line-number="1111"></td>
        <td id="LC1111" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>animator</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Animator</span>(<span class=pl-s1>xlabel</span><span class=pl-c1>=</span><span class=pl-s>&#39;epoch&#39;</span>, <span class=pl-s1>ylabel</span><span class=pl-c1>=</span><span class=pl-s>&#39;loss&#39;</span>,</td>
      </tr>
      <tr>
        <td id="L1112" class="blob-num js-line-number" data-line-number="1112"></td>
        <td id="LC1112" class="blob-code blob-code-inner js-file-line">                            <span class=pl-s1>xlim</span><span class=pl-c1>=</span>[<span class=pl-c1>0</span>, <span class=pl-s1>num_epochs</span>], <span class=pl-s1>ylim</span><span class=pl-c1>=</span>[<span class=pl-c1>0.22</span>, <span class=pl-c1>0.35</span>])</td>
      </tr>
      <tr>
        <td id="L1113" class="blob-num js-line-number" data-line-number="1113"></td>
        <td id="LC1113" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>n</span>, <span class=pl-s1>timer</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span>, <span class=pl-s1>d2l</span>.<span class=pl-v>Timer</span>()</td>
      </tr>
      <tr>
        <td id="L1114" class="blob-num js-line-number" data-line-number="1114"></td>
        <td id="LC1114" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>_</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-s1>num_epochs</span>):</td>
      </tr>
      <tr>
        <td id="L1115" class="blob-num js-line-number" data-line-number="1115"></td>
        <td id="LC1115" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-v>X</span>, <span class=pl-s1>y</span> <span class=pl-c1>in</span> <span class=pl-s1>data_iter</span>:</td>
      </tr>
      <tr>
        <td id="L1116" class="blob-num js-line-number" data-line-number="1116"></td>
        <td id="LC1116" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>l</span> <span class=pl-c1>=</span> <span class=pl-en>loss</span>(<span class=pl-en>net</span>(<span class=pl-v>X</span>), <span class=pl-s1>y</span>).<span class=pl-en>mean</span>()</td>
      </tr>
      <tr>
        <td id="L1117" class="blob-num js-line-number" data-line-number="1117"></td>
        <td id="LC1117" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>l</span>.<span class=pl-en>backward</span>()</td>
      </tr>
      <tr>
        <td id="L1118" class="blob-num js-line-number" data-line-number="1118"></td>
        <td id="LC1118" class="blob-code blob-code-inner js-file-line">            <span class=pl-en>trainer_fn</span>([<span class=pl-s1>w</span>, <span class=pl-s1>b</span>], <span class=pl-s1>states</span>, <span class=pl-s1>hyperparams</span>)</td>
      </tr>
      <tr>
        <td id="L1119" class="blob-num js-line-number" data-line-number="1119"></td>
        <td id="LC1119" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>n</span> <span class=pl-c1>+=</span> <span class=pl-v>X</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>0</span>]</td>
      </tr>
      <tr>
        <td id="L1120" class="blob-num js-line-number" data-line-number="1120"></td>
        <td id="LC1120" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>if</span> <span class=pl-s1>n</span> <span class=pl-c1>%</span> <span class=pl-c1>200</span> <span class=pl-c1>==</span> <span class=pl-c1>0</span>:</td>
      </tr>
      <tr>
        <td id="L1121" class="blob-num js-line-number" data-line-number="1121"></td>
        <td id="LC1121" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>timer</span>.<span class=pl-en>stop</span>()</td>
      </tr>
      <tr>
        <td id="L1122" class="blob-num js-line-number" data-line-number="1122"></td>
        <td id="LC1122" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>animator</span>.<span class=pl-en>add</span>(<span class=pl-s1>n</span><span class=pl-c1>/</span><span class=pl-v>X</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>0</span>]<span class=pl-c1>/</span><span class=pl-en>len</span>(<span class=pl-s1>data_iter</span>),</td>
      </tr>
      <tr>
        <td id="L1123" class="blob-num js-line-number" data-line-number="1123"></td>
        <td id="LC1123" class="blob-code blob-code-inner js-file-line">                             (<span class=pl-s1>d2l</span>.<span class=pl-en>evaluate_loss</span>(<span class=pl-s1>net</span>, <span class=pl-s1>data_iter</span>, <span class=pl-s1>loss</span>),))</td>
      </tr>
      <tr>
        <td id="L1124" class="blob-num js-line-number" data-line-number="1124"></td>
        <td id="LC1124" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>timer</span>.<span class=pl-en>start</span>()</td>
      </tr>
      <tr>
        <td id="L1125" class="blob-num js-line-number" data-line-number="1125"></td>
        <td id="LC1125" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>f&#39;loss: <span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>animator</span>.<span class=pl-v>Y</span>[<span class=pl-c1>0</span>][<span class=pl-c1>-</span><span class=pl-c1>1</span>]:.3f<span class=pl-kos>}</span></span>, <span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>timer</span>.<span class=pl-en>avg</span>():.3f<span class=pl-kos>}</span></span> sec/epoch&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L1126" class="blob-num js-line-number" data-line-number="1126"></td>
        <td id="LC1126" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>timer</span>.<span class=pl-en>cumsum</span>(), <span class=pl-s1>animator</span>.<span class=pl-v>Y</span>[<span class=pl-c1>0</span>]</td>
      </tr>
      <tr>
        <td id="L1127" class="blob-num js-line-number" data-line-number="1127"></td>
        <td id="LC1127" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1128" class="blob-num js-line-number" data-line-number="1128"></td>
        <td id="LC1128" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1129" class="blob-num js-line-number" data-line-number="1129"></td>
        <td id="LC1129" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Defined in file: ./chapter_optimization/minibatch-sgd.md</span></td>
      </tr>
      <tr>
        <td id="L1130" class="blob-num js-line-number" data-line-number="1130"></td>
        <td id="LC1130" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>train_concise_ch11</span>(<span class=pl-s1>trainer_fn</span>, <span class=pl-s1>hyperparams</span>, <span class=pl-s1>data_iter</span>, <span class=pl-s1>num_epochs</span><span class=pl-c1>=</span><span class=pl-c1>4</span>):</td>
      </tr>
      <tr>
        <td id="L1131" class="blob-num js-line-number" data-line-number="1131"></td>
        <td id="LC1131" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Initialization</span></td>
      </tr>
      <tr>
        <td id="L1132" class="blob-num js-line-number" data-line-number="1132"></td>
        <td id="LC1132" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>net</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>Sequential</span>(<span class=pl-s1>nn</span>.<span class=pl-v>Linear</span>(<span class=pl-c1>5</span>, <span class=pl-c1>1</span>))</td>
      </tr>
      <tr>
        <td id="L1133" class="blob-num js-line-number" data-line-number="1133"></td>
        <td id="LC1133" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>def</span> <span class=pl-en>init_weights</span>(<span class=pl-s1>m</span>):</td>
      </tr>
      <tr>
        <td id="L1134" class="blob-num js-line-number" data-line-number="1134"></td>
        <td id="LC1134" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>if</span> <span class=pl-en>type</span>(<span class=pl-s1>m</span>) <span class=pl-c1>==</span> <span class=pl-s1>nn</span>.<span class=pl-v>Linear</span>:</td>
      </tr>
      <tr>
        <td id="L1135" class="blob-num js-line-number" data-line-number="1135"></td>
        <td id="LC1135" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>torch</span>.<span class=pl-s1>nn</span>.<span class=pl-s1>init</span>.<span class=pl-en>normal_</span>(<span class=pl-s1>m</span>.<span class=pl-s1>weight</span>, <span class=pl-s1>std</span><span class=pl-c1>=</span><span class=pl-c1>0.01</span>)</td>
      </tr>
      <tr>
        <td id="L1136" class="blob-num js-line-number" data-line-number="1136"></td>
        <td id="LC1136" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>net</span>.<span class=pl-en>apply</span>(<span class=pl-s1>init_weights</span>)</td>
      </tr>
      <tr>
        <td id="L1137" class="blob-num js-line-number" data-line-number="1137"></td>
        <td id="LC1137" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L1138" class="blob-num js-line-number" data-line-number="1138"></td>
        <td id="LC1138" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>optimizer</span> <span class=pl-c1>=</span> <span class=pl-en>trainer_fn</span>(<span class=pl-s1>net</span>.<span class=pl-en>parameters</span>(), <span class=pl-c1>**</span><span class=pl-s1>hyperparams</span>)</td>
      </tr>
      <tr>
        <td id="L1139" class="blob-num js-line-number" data-line-number="1139"></td>
        <td id="LC1139" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L1140" class="blob-num js-line-number" data-line-number="1140"></td>
        <td id="LC1140" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>loss</span> <span class=pl-c1>=</span> <span class=pl-s1>nn</span>.<span class=pl-v>MSELoss</span>()</td>
      </tr>
      <tr>
        <td id="L1141" class="blob-num js-line-number" data-line-number="1141"></td>
        <td id="LC1141" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Note: L2 Loss = 1/2 * MSE Loss. PyTorch has MSE Loss which is slightly</span></td>
      </tr>
      <tr>
        <td id="L1142" class="blob-num js-line-number" data-line-number="1142"></td>
        <td id="LC1142" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># different from MXNet&#39;s L2Loss by a factor of 2. Hence we halve the loss </span></td>
      </tr>
      <tr>
        <td id="L1143" class="blob-num js-line-number" data-line-number="1143"></td>
        <td id="LC1143" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># value to get L2Loss in PyTorch.</span></td>
      </tr>
      <tr>
        <td id="L1144" class="blob-num js-line-number" data-line-number="1144"></td>
        <td id="LC1144" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L1145" class="blob-num js-line-number" data-line-number="1145"></td>
        <td id="LC1145" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>animator</span> <span class=pl-c1>=</span> <span class=pl-s1>d2l</span>.<span class=pl-v>Animator</span>(<span class=pl-s1>xlabel</span><span class=pl-c1>=</span><span class=pl-s>&#39;epoch&#39;</span>, <span class=pl-s1>ylabel</span><span class=pl-c1>=</span><span class=pl-s>&#39;loss&#39;</span>,</td>
      </tr>
      <tr>
        <td id="L1146" class="blob-num js-line-number" data-line-number="1146"></td>
        <td id="LC1146" class="blob-code blob-code-inner js-file-line">                            <span class=pl-s1>xlim</span><span class=pl-c1>=</span>[<span class=pl-c1>0</span>, <span class=pl-s1>num_epochs</span>], <span class=pl-s1>ylim</span><span class=pl-c1>=</span>[<span class=pl-c1>0.22</span>, <span class=pl-c1>0.35</span>])</td>
      </tr>
      <tr>
        <td id="L1147" class="blob-num js-line-number" data-line-number="1147"></td>
        <td id="LC1147" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>n</span>, <span class=pl-s1>timer</span> <span class=pl-c1>=</span> <span class=pl-c1>0</span>, <span class=pl-s1>d2l</span>.<span class=pl-v>Timer</span>()</td>
      </tr>
      <tr>
        <td id="L1148" class="blob-num js-line-number" data-line-number="1148"></td>
        <td id="LC1148" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>_</span> <span class=pl-c1>in</span> <span class=pl-en>range</span>(<span class=pl-s1>num_epochs</span>):</td>
      </tr>
      <tr>
        <td id="L1149" class="blob-num js-line-number" data-line-number="1149"></td>
        <td id="LC1149" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-v>X</span>, <span class=pl-s1>y</span> <span class=pl-c1>in</span> <span class=pl-s1>data_iter</span>:</td>
      </tr>
      <tr>
        <td id="L1150" class="blob-num js-line-number" data-line-number="1150"></td>
        <td id="LC1150" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>optimizer</span>.<span class=pl-en>zero_grad</span>()</td>
      </tr>
      <tr>
        <td id="L1151" class="blob-num js-line-number" data-line-number="1151"></td>
        <td id="LC1151" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>out</span> <span class=pl-c1>=</span> <span class=pl-en>net</span>(<span class=pl-v>X</span>)</td>
      </tr>
      <tr>
        <td id="L1152" class="blob-num js-line-number" data-line-number="1152"></td>
        <td id="LC1152" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>y</span> <span class=pl-c1>=</span> <span class=pl-s1>y</span>.<span class=pl-en>reshape</span>(<span class=pl-s1>out</span>.<span class=pl-s1>shape</span>)</td>
      </tr>
      <tr>
        <td id="L1153" class="blob-num js-line-number" data-line-number="1153"></td>
        <td id="LC1153" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>l</span> <span class=pl-c1>=</span> <span class=pl-en>loss</span>(<span class=pl-s1>out</span>, <span class=pl-s1>y</span>)<span class=pl-c1>/</span><span class=pl-c1>2</span></td>
      </tr>
      <tr>
        <td id="L1154" class="blob-num js-line-number" data-line-number="1154"></td>
        <td id="LC1154" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>l</span>.<span class=pl-en>backward</span>()</td>
      </tr>
      <tr>
        <td id="L1155" class="blob-num js-line-number" data-line-number="1155"></td>
        <td id="LC1155" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>optimizer</span>.<span class=pl-en>step</span>()</td>
      </tr>
      <tr>
        <td id="L1156" class="blob-num js-line-number" data-line-number="1156"></td>
        <td id="LC1156" class="blob-code blob-code-inner js-file-line">            <span class=pl-s1>n</span> <span class=pl-c1>+=</span> <span class=pl-v>X</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>0</span>]</td>
      </tr>
      <tr>
        <td id="L1157" class="blob-num js-line-number" data-line-number="1157"></td>
        <td id="LC1157" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>if</span> <span class=pl-s1>n</span> <span class=pl-c1>%</span> <span class=pl-c1>200</span> <span class=pl-c1>==</span> <span class=pl-c1>0</span>:</td>
      </tr>
      <tr>
        <td id="L1158" class="blob-num js-line-number" data-line-number="1158"></td>
        <td id="LC1158" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>timer</span>.<span class=pl-en>stop</span>()</td>
      </tr>
      <tr>
        <td id="L1159" class="blob-num js-line-number" data-line-number="1159"></td>
        <td id="LC1159" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>animator</span>.<span class=pl-en>add</span>(<span class=pl-s1>n</span><span class=pl-c1>/</span><span class=pl-v>X</span>.<span class=pl-s1>shape</span>[<span class=pl-c1>0</span>]<span class=pl-c1>/</span><span class=pl-en>len</span>(<span class=pl-s1>data_iter</span>),</td>
      </tr>
      <tr>
        <td id="L1160" class="blob-num js-line-number" data-line-number="1160"></td>
        <td id="LC1160" class="blob-code blob-code-inner js-file-line">                             (<span class=pl-s1>d2l</span>.<span class=pl-en>evaluate_loss</span>(<span class=pl-s1>net</span>, <span class=pl-s1>data_iter</span>, <span class=pl-s1>loss</span>)<span class=pl-c1>/</span><span class=pl-c1>2</span>,))</td>
      </tr>
      <tr>
        <td id="L1161" class="blob-num js-line-number" data-line-number="1161"></td>
        <td id="LC1161" class="blob-code blob-code-inner js-file-line">                <span class=pl-s1>timer</span>.<span class=pl-en>start</span>()</td>
      </tr>
      <tr>
        <td id="L1162" class="blob-num js-line-number" data-line-number="1162"></td>
        <td id="LC1162" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>f&#39;loss: <span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>animator</span>.<span class=pl-v>Y</span>[<span class=pl-c1>0</span>][<span class=pl-c1>-</span><span class=pl-c1>1</span>]:.3f<span class=pl-kos>}</span></span>, <span class=pl-s1><span class=pl-kos>{</span><span class=pl-s1>timer</span>.<span class=pl-en>avg</span>():.3f<span class=pl-kos>}</span></span> sec/epoch&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L1163" class="blob-num js-line-number" data-line-number="1163"></td>
        <td id="LC1163" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1164" class="blob-num js-line-number" data-line-number="1164"></td>
        <td id="LC1164" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1165" class="blob-num js-line-number" data-line-number="1165"></td>
        <td id="LC1165" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Alias defined in config.ini</span></td>
      </tr>
      <tr>
        <td id="L1166" class="blob-num js-line-number" data-line-number="1166"></td>
        <td id="LC1166" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1167" class="blob-num js-line-number" data-line-number="1167"></td>
        <td id="LC1167" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1168" class="blob-num js-line-number" data-line-number="1168"></td>
        <td id="LC1168" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>ones</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>ones</span></td>
      </tr>
      <tr>
        <td id="L1169" class="blob-num js-line-number" data-line-number="1169"></td>
        <td id="LC1169" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>zeros</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>zeros</span></td>
      </tr>
      <tr>
        <td id="L1170" class="blob-num js-line-number" data-line-number="1170"></td>
        <td id="LC1170" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>tensor</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>tensor</span></td>
      </tr>
      <tr>
        <td id="L1171" class="blob-num js-line-number" data-line-number="1171"></td>
        <td id="LC1171" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>arange</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>arange</span></td>
      </tr>
      <tr>
        <td id="L1172" class="blob-num js-line-number" data-line-number="1172"></td>
        <td id="LC1172" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>meshgrid</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>meshgrid</span></td>
      </tr>
      <tr>
        <td id="L1173" class="blob-num js-line-number" data-line-number="1173"></td>
        <td id="LC1173" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>sin</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>sin</span></td>
      </tr>
      <tr>
        <td id="L1174" class="blob-num js-line-number" data-line-number="1174"></td>
        <td id="LC1174" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>sinh</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>sinh</span></td>
      </tr>
      <tr>
        <td id="L1175" class="blob-num js-line-number" data-line-number="1175"></td>
        <td id="LC1175" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>cos</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>cos</span></td>
      </tr>
      <tr>
        <td id="L1176" class="blob-num js-line-number" data-line-number="1176"></td>
        <td id="LC1176" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>cosh</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>cosh</span></td>
      </tr>
      <tr>
        <td id="L1177" class="blob-num js-line-number" data-line-number="1177"></td>
        <td id="LC1177" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>tanh</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>tanh</span></td>
      </tr>
      <tr>
        <td id="L1178" class="blob-num js-line-number" data-line-number="1178"></td>
        <td id="LC1178" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>linspace</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>linspace</span></td>
      </tr>
      <tr>
        <td id="L1179" class="blob-num js-line-number" data-line-number="1179"></td>
        <td id="LC1179" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>exp</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>exp</span></td>
      </tr>
      <tr>
        <td id="L1180" class="blob-num js-line-number" data-line-number="1180"></td>
        <td id="LC1180" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>log</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>log</span></td>
      </tr>
      <tr>
        <td id="L1181" class="blob-num js-line-number" data-line-number="1181"></td>
        <td id="LC1181" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>normal</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>normal</span></td>
      </tr>
      <tr>
        <td id="L1182" class="blob-num js-line-number" data-line-number="1182"></td>
        <td id="LC1182" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>matmul</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>matmul</span></td>
      </tr>
      <tr>
        <td id="L1183" class="blob-num js-line-number" data-line-number="1183"></td>
        <td id="LC1183" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>int32</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>int32</span></td>
      </tr>
      <tr>
        <td id="L1184" class="blob-num js-line-number" data-line-number="1184"></td>
        <td id="LC1184" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>float32</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>float32</span></td>
      </tr>
      <tr>
        <td id="L1185" class="blob-num js-line-number" data-line-number="1185"></td>
        <td id="LC1185" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>concat</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>cat</span></td>
      </tr>
      <tr>
        <td id="L1186" class="blob-num js-line-number" data-line-number="1186"></td>
        <td id="LC1186" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>stack</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>stack</span></td>
      </tr>
      <tr>
        <td id="L1187" class="blob-num js-line-number" data-line-number="1187"></td>
        <td id="LC1187" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>abs</span> <span class=pl-c1>=</span> <span class=pl-s1>torch</span>.<span class=pl-s1>abs</span></td>
      </tr>
      <tr>
        <td id="L1188" class="blob-num js-line-number" data-line-number="1188"></td>
        <td id="LC1188" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>numpy</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span> <span class=pl-s1>x</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>: <span class=pl-s1>x</span>.<span class=pl-en>detach</span>().<span class=pl-en>numpy</span>(<span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L1189" class="blob-num js-line-number" data-line-number="1189"></td>
        <td id="LC1189" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>size</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span> <span class=pl-s1>x</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>: <span class=pl-s1>x</span>.<span class=pl-en>numel</span>(<span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L1190" class="blob-num js-line-number" data-line-number="1190"></td>
        <td id="LC1190" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>reshape</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span> <span class=pl-s1>x</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>: <span class=pl-s1>x</span>.<span class=pl-en>reshape</span>(<span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L1191" class="blob-num js-line-number" data-line-number="1191"></td>
        <td id="LC1191" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>to</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span> <span class=pl-s1>x</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>: <span class=pl-s1>x</span>.<span class=pl-en>to</span>(<span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L1192" class="blob-num js-line-number" data-line-number="1192"></td>
        <td id="LC1192" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>reduce_sum</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span> <span class=pl-s1>x</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>: <span class=pl-s1>x</span>.<span class=pl-en>sum</span>(<span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L1193" class="blob-num js-line-number" data-line-number="1193"></td>
        <td id="LC1193" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>argmax</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span> <span class=pl-s1>x</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>: <span class=pl-s1>x</span>.<span class=pl-en>argmax</span>(<span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L1194" class="blob-num js-line-number" data-line-number="1194"></td>
        <td id="LC1194" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>astype</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span> <span class=pl-s1>x</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>: <span class=pl-s1>x</span>.<span class=pl-en>type</span>(<span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
      <tr>
        <td id="L1195" class="blob-num js-line-number" data-line-number="1195"></td>
        <td id="LC1195" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>transpose</span> <span class=pl-c1>=</span> <span class=pl-k>lambda</span> <span class=pl-s1>x</span>, <span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>: <span class=pl-s1>x</span>.<span class=pl-en>t</span>(<span class=pl-c1>*</span><span class=pl-s1>args</span>, <span class=pl-c1>**</span><span class=pl-s1>kwargs</span>)</td>
      </tr>
</table>

  <details class="details-reset details-overlay BlobToolbar position-absolute js-file-line-actions dropdown d-none" aria-hidden="true">
    <summary class="btn-octicon ml-0 px-2 p-0 bg-white border border-gray-dark rounded-1" aria-label="Inline file action toolbar">
      <svg class="octicon octicon-kebab-horizontal" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="M8 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM1.5 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm13 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3z"></path></svg>
    </summary>
    <details-menu>
      <ul class="BlobToolbar-dropdown dropdown-menu dropdown-menu-se mt-2" style="width:185px">
        <li>
          <clipboard-copy role="menuitem" class="dropdown-item" id="js-copy-lines" style="cursor:pointer;">
            Copy lines
          </clipboard-copy>
        </li>
        <li>
          <clipboard-copy role="menuitem" class="dropdown-item" id="js-copy-permalink" style="cursor:pointer;">
            Copy permalink
          </clipboard-copy>
        </li>
        <li><a class="dropdown-item js-update-url-with-hash" id="js-view-git-blame" role="menuitem" href="/d2l-ai/d2l-en/blame/9607f62e2bb336e25f102347d44907e3f69e567a/d2l/torch.py">View git blame</a></li>
          <li><a class="dropdown-item" id="js-new-issue" role="menuitem" href="/d2l-ai/d2l-en/issues/new">Reference in new issue</a></li>
      </ul>
    </details-menu>
  </details>

  </div>

    </div>

  


  <details class="details-reset details-overlay details-overlay-dark" id="jumpto-line-details-dialog">
    <summary data-hotkey="l" aria-label="Jump to line"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast linejump" aria-label="Jump to line">
      <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-jump-to-line-form Box-body d-flex" action="" accept-charset="UTF-8" method="get">
        <input class="form-control flex-auto mr-3 linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" aria-label="Jump to line" autofocus>
        <button type="submit" class="btn" data-close-dialog>Go</button>
</form>    </details-dialog>
  </details>

    <div class="Popover anim-scale-in js-tagsearch-popover"
     hidden
     data-tagsearch-url="/d2l-ai/d2l-en/find-definition"
     data-tagsearch-ref="master"
     data-tagsearch-path="d2l/torch.py"
     data-tagsearch-lang="Python"
     data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.click_on_symbol&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;click_on_symbol&quot;,&quot;repository_id&quot;:152166877,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py&quot;,&quot;user_id&quot;:null}}"
     data-hydro-click-hmac="6fb2a9b896c93519306a7a4715cb84a340aff64550c1141d352cd62c167dbfab">
  <div class="Popover-message Popover-message--large Popover-message--top-left TagsearchPopover mt-1 mb-4 mx-auto Box box-shadow-large">
    <div class="TagsearchPopover-content js-tagsearch-popover-content overflow-auto" style="will-change:transform;">
    </div>
  </div>
</div>




  </div>
</div>

    </main>
  </div>
  

  </div>

        
<div class="footer container-xl width-full p-responsive" role="contentinfo">
  <div class="position-relative d-flex flex-row-reverse flex-lg-row flex-wrap flex-lg-nowrap flex-justify-center flex-lg-justify-between pt-6 pb-2 mt-6 f6 text-gray border-top border-gray-light ">
    <ul class="list-style-none d-flex flex-wrap col-12 col-lg-5 flex-justify-center flex-lg-justify-between mb-2 mb-lg-0">
      <li class="mr-3 mr-lg-0">&copy; 2020 GitHub, Inc.</li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to terms, text:terms" href="https://github.com/site/terms">Terms</a></li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to privacy, text:privacy" href="https://github.com/site/privacy">Privacy</a></li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to security, text:security" href="https://github.com/security">Security</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://githubstatus.com/" data-ga-click="Footer, go to status, text:status">Status</a></li>
        <li><a data-ga-click="Footer, go to help, text:help" href="https://help.github.com">Help</a></li>

    </ul>

    <a aria-label="Homepage" title="GitHub" class="footer-octicon d-none d-lg-block mx-lg-4" href="https://github.com">
      <svg height="24" class="octicon octicon-mark-github" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
</a>
   <ul class="list-style-none d-flex flex-wrap col-12 col-lg-5 flex-justify-center flex-lg-justify-between mb-2 mb-lg-0">
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to contact, text:contact" href="https://github.com/contact">Contact GitHub</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://github.com/pricing" data-ga-click="Footer, go to Pricing, text:Pricing">Pricing</a></li>
      <li class="mr-3 mr-lg-0"><a href="https://developer.github.com" data-ga-click="Footer, go to api, text:api">API</a></li>
      <li class="mr-3 mr-lg-0"><a href="https://training.github.com" data-ga-click="Footer, go to training, text:training">Training</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://github.blog" data-ga-click="Footer, go to blog, text:blog">Blog</a></li>
        <li><a data-ga-click="Footer, go to about, text:about" href="https://github.com/about">About</a></li>
    </ul>
  </div>
  <div class="d-flex flex-justify-center pb-6">
    <span class="f6 text-gray-light"></span>
  </div>
</div>



  <div id="ajax-error-message" class="ajax-error-message flash flash-error">
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.22 1.754a.25.25 0 00-.44 0L1.698 13.132a.25.25 0 00.22.368h12.164a.25.25 0 00.22-.368L8.22 1.754zm-1.763-.707c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0114.082 15H1.918a1.75 1.75 0 01-1.543-2.575L6.457 1.047zM9 11a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.25a.75.75 0 00-1.5 0v2.5a.75.75 0 001.5 0v-2.5z"></path></svg>
    <button type="button" class="flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
    </button>
    You can’t perform that action at this time.
  </div>


    <script crossorigin="anonymous" async="async" integrity="sha512-bn/3rKJzBl2H64K38R8KaVcT26vKK7BJQC59lwYc+9fjlHzmy0fwh+hzBtsgTdhIi13dxjzNKWhdSN8WTM9qUw==" type="application/javascript" id="js-conditional-compat" data-src="https://github.githubassets.com/assets/compat-bootstrap-6e7ff7ac.js"></script>
    <script crossorigin="anonymous" integrity="sha512-xhGdF/fp8V2bfPp/iXtkF1HCVnHsIFAYUt+IB/WRnZ4mUihf3I+WFIktIzRY9NbilzMPP13nw5Uo3kIA53t4gQ==" type="application/javascript" src="https://github.githubassets.com/assets/environment-bootstrap-c6119d17.js"></script>
    <script crossorigin="anonymous" async="async" integrity="sha512-Ncm8EEN+jdJqci1dTZ4/BCY+ueKmNOQiPQj6rcX2/6qKIdf1hJqQZnJJEjoFj18nEkD/hus03JOBTjmHB3jNMg==" type="application/javascript" src="https://github.githubassets.com/assets/vendor-35c9bc10.js"></script>
    <script crossorigin="anonymous" async="async" integrity="sha512-VvafeXTzUvpmD50UYlFogaWkqft7Ft6nZDrrMYnOt6teK7K0jdsMTPK15IoB0QyD4urDH5nfruPyckKuuUxaWQ==" type="application/javascript" src="https://github.githubassets.com/assets/frameworks-56f69f79.js"></script>
    
    <script crossorigin="anonymous" async="async" integrity="sha512-ykVqKskYBMgqGN/dZ+ObXjsBMy2bZCrHswYpUilhUd6eZyJ1itSqiqQ1NYgmdL/P36y7Gotcjv9IKpd1TghIaQ==" type="application/javascript" src="https://github.githubassets.com/assets/github-bootstrap-ca456a2a.js"></script>
    
      <script crossorigin="anonymous" async="async" integrity="sha512-TjmDUfspgN28WRWfc01tOL0BFS8pI/TAi8TQ665TcA3jG1C3QgfFu0YKa32Z03rlEL8dukbsy+amwBzgGyjESQ==" type="application/javascript" data-module-id="./Sortable.js" data-src="https://github.githubassets.com/assets/Sortable-4e398351.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-4GcSWGoe36+BoWho4gtJcByZe8j43w+lt2/PDe3rmBxRVSgD29YipDwuIywe8fvOd2b2CszBqaPGxSznUtE3Xg==" type="application/javascript" data-module-id="./drag-drop.js" data-src="https://github.githubassets.com/assets/drag-drop-e0671258.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-MttsTK6LRLl4AiQlvZ8MfNsDe0brgs9ubvDV/Ck6sVM+MnjEn+6tfF4hS8ENrXG1v+lBHFWmAS5j43gum0xsXw==" type="application/javascript" data-module-id="./gist-vendor.js" data-src="https://github.githubassets.com/assets/gist-vendor-32db6c4c.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-iLuC2weaJqL9mYAud2WDWjhd8cJe8dXVxw2KhCH2Rnj6WJvTzlZRmvTtL09wNWX6nRze/TDaQ7gq7BFLchaDYg==" type="application/javascript" data-module-id="./image-crop-element-loader.js" data-src="https://github.githubassets.com/assets/image-crop-element-loader-88bb82db.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-SzrmrFC6Li3booxqs0mRixus2NKXsmDzy81YKIdwyd4llzBUojVrUd87DAv4Gm3LlROU45cG5C6/noBz+/exMA==" type="application/javascript" data-module-id="./profile-pins-element.js" data-src="https://github.githubassets.com/assets/profile-pins-element-4b3ae6ac.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-qECv/jhsvLFN77eGNu0cjMR2+zvAlLyhQVTnmayJc5OLZoxMLjQZxZW1hK/dhcYro6Wec/aiF21HYf2N5OilYQ==" type="application/javascript" data-module-id="./randomColor.js" data-src="https://github.githubassets.com/assets/randomColor-a840affe.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-45AwxR1TB7Z8BL0dnYOrINtveNF4Du3OaUdubEdbdfYrswXjalLxzIFenU8e6CVEoL6pHMHLLzXRPFokwAWmsw==" type="application/javascript" data-module-id="./tweetsodium.js" data-src="https://github.githubassets.com/assets/tweetsodium-e39030c5.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-pyCjN7R7IeimPxv2cWSgc7Jovuu5uCqEV8sJthsjO8DQMX2oXoLnPhzyph1U+GpMuScf+xfhNbpUQrRXohaeLw==" type="application/javascript" data-module-id="./user-status-submit.js" data-src="https://github.githubassets.com/assets/user-status-submit-a720a337.js"></script>
    
    
  <div class="js-stale-session-flash flash flash-warn flash-banner" hidden
    >
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.22 1.754a.25.25 0 00-.44 0L1.698 13.132a.25.25 0 00.22.368h12.164a.25.25 0 00.22-.368L8.22 1.754zm-1.763-.707c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0114.082 15H1.918a1.75 1.75 0 01-1.543-2.575L6.457 1.047zM9 11a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.25a.75.75 0 00-1.5 0v2.5a.75.75 0 001.5 0v-2.5z"></path></svg>
    <span class="js-stale-session-flash-signed-in" hidden>You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
    <span class="js-stale-session-flash-signed-out" hidden>You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
  </div>
  <template id="site-details-dialog">
  <details class="details-reset details-overlay details-overlay-dark lh-default text-gray-dark hx_rsm" open>
    <summary role="button" aria-label="Close dialog"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast hx_rsm-dialog hx_rsm-modal">
      <button class="Box-btn-octicon m-0 btn-octicon position-absolute right-0 top-0" type="button" aria-label="Close dialog" data-close-dialog>
        <svg class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
      </button>
      <div class="octocat-spinner my-6 js-details-dialog-spinner"></div>
    </details-dialog>
  </details>
</template>

  <div class="Popover js-hovercard-content position-absolute" style="display: none; outline: none;" tabindex="0">
  <div class="Popover-message Popover-message--bottom-left Popover-message--large Box box-shadow-large" style="width:360px;">
  </div>
</div>


  </body>
</html>

