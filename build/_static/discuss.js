$(document).ready(function () {
    var discuss = $("#discuss")
    var topic_id = discuss.attr("topic_id");
    discuss.html('<div id="discourse-comments"></div>');
    DiscourseEmbed = { discourseUrl: 'https://discuss.gluon.ai/', topicId: topic_id };
    (function() {
        var d = document.createElement('script'); d.type = 'text/javascript';
        d.async = true;
        d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
        (document.getElementsByTagName('head')[0] ||
         document.getElementsByTagName('body')[0]).appendChild(d);
    })();
});
