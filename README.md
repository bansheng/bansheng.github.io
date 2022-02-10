<!--
 * @Author: dingyadong
 * @Date: 2022-02-10 17:20:19
 * @LastEditTime: 2022-02-10 17:53:17
 * @LastEditors: dingyadong
 * @Description:
 * @FilePath: /003-github.io/README.md
-->

# 版本历史

## V0.1

```swig
/Users/admin/02-privateDir/003-github.io/themes/next/layout/_partials/footer.swig
# 添加评论

{% if theme.footer.counter and theme.footer.counter.enable %}
  <span id="busuanzi_container_site_pv">总访问量<span id="busuanzi_value_site_pv"></span>次</span>
  <span class="post-meta-divider">|</span>
  <span id="busuanzi_container_site_uv">总访客数<span id="busuanzi_value_site_uv"></span>人</span>
{% endif %}
```

```yml
/Users/admin/02-privateDir/003-github.io/themes/next/_config.yml

menu:
  home: /|| fa fa-home
  tags: /tags/|| fa fa-tags
  categories: /categories/|| fa fa-th
  archives: /archives/|| fa fa-archive
  about: /about/|| fa fa-user
  contact: /contact/|| fa fa-heartbeat
  sitemap: /sitemap.xml|| fa fa-sitemap
  #schedule: /schedule/ || fa fa-calendar
  # commonweal: /404/|| fa fa-heartbeat

social:
  GitHub: https://github.com/bansheng|| fab fa-github
  E-Mail: mailto:dyadongcs@gmail.com|| fa fa-envelope
  Zhihu:  https://www.zhihu.com/people/lemon-dong|| fab fa-zhihu

related_posts:
  enable: false
  title: 猜你喜欢 # 自定义标题名字
  display_in_home: true # 首页是否增加
  params:
    maxCount: 5 # 最多推荐几个
    PPMixingRate: 0.4 # 同时推荐火热和相关，两者比率，不能为0
    #isDate: false # 文章时间
    #isImage: false # 文章配图
    #isExcerpt: false # 文章摘要

pjax: true

auto_excerpt:
  enable: true
  length: 150

waline:
  enable: true #是否开启
  serverURL: waline-server-bansheng.vercel.app # Waline #服务端地址，我们这里就是上面部署的 Vercel 地址
  placeholder: 请文明评论呀~ # #评论框的默认文字
  avatar: wavatar # 头像风格
  meta: [nick, mail, link] # 自定义评论框上面的三个输入框的内容
  pageSize: 10 # 评论数量多少时显示分页
  lang: zh-cn # 语言, 可选值: en, zh-cn
  # Warning: 不要同时启用 `waline.visitor` 以及 `leancloud_visitors`.
  visitor: false # 文章阅读统计
  comment_count: true # 如果为 false , 评论数量只会在当前评论页面显示, 主页则不显示
  requiredFields: [nick] # 设置用户评论时必填的信息，[nick,mail]: [nick] | [nick, mail]
  libUrl: # Set custom library cdn url

```

## 0.2

1. 切换next v8.0
2. 添加waline 评论
3. 添加访客人数
4. set fancybox: true
5. local_search:
  enable: true
