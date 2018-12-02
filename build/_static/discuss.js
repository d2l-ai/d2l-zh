$(document).ready(function () {
    $('h2').each(function(){
        if ($(this).text().indexOf("扫码") != -1) {
            var url = $(this).find('a').attr('href');
            var tokens = url.split('/');
            var topic_id = tokens[tokens.length-1];
            $(this).html('<h2>参与讨论</h2>');
            $(this).parent().append('<div id="discourse-comments"></div>');

            $('a').each(function(){
                if ($(this).text().indexOf("扫码直达讨论区") != -1) {
                    $(this).text('参与讨论');
                }
            });

            $('img').each(function(){
                if ($(this).attr('src').indexOf("qr_") != -1) {
                    $(this).hide();
                }
            });

            DiscourseEmbed = { discourseUrl: 'https://discuss.gluon.ai/', topicId: topic_id };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript';
                d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] ||
                 document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        }
    });

});
