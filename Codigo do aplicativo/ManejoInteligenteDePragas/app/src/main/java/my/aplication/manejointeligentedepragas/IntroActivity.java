package my.aplication.manejointeligentedepragas;



import android.content.Intent;
import android.content.SharedPreferences;

//import android.support.design.widget.TabLayout;
import com.google.android.material.tabs.TabLayout;

import androidx.appcompat.app.AppCompatActivity;
import androidx.viewpager.widget.ViewPager;

import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.Button;

import com.example.manejointeligentedepragas.R;

import java.util.ArrayList;
import java.util.List;

public class IntroActivity extends AppCompatActivity {


    private ViewPager screenPager;
    IntroViewPagerAdapter introViewPagerAdapter;
    TabLayout tabindicator;
    Button btnNext;
    Button getStarted;
    int position = 0;
    Animation btnAnim;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //make the activty full screen
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);


        //when this activity is about to be launch we need to check if it's opened before or not
        if(restorePrefData()){
            Intent main = new Intent(getApplicationContext(), MainActivity.class);
            startActivity(main);
        }

        //adding content
        setContentView(R.layout.activity_intro);

        //hiding the action bar





        //ini views
        btnNext = findViewById(R.id.buttonIntro);
        getStarted = findViewById(R.id.btnGetStarted);
        tabindicator =findViewById(R.id.tab_indicator);
        btnAnim = AnimationUtils.loadAnimation(getApplicationContext(), R.anim.btn_animation);

        //Fill list screen
        final List<ScreenItem> mList = new ArrayList<>();
        mList.add(new ScreenItem("Bem-vindo ao MIP²!", "Estamos felizes que está por aqui. Vamos apresentar brevemente as principais funções do aplicativo, aproveite.", R.drawable.semfundoo));
        mList.add(new ScreenItem("Entrar e menu", "Para acessar as funções principais do aplicativo, cadastre-se e entre. O menu estará disponível em todas as telas, clique no ícone indicado para abrí-lo. É pelo menu que você terá acesso às informações disponibilizadas pelo aplicativo.", R.drawable.segundatela));
        mList.add(new ScreenItem("Propriedades", "Essa é a tela de propriedades, nela será possível adicionar, visualizar, editar ou excluir uma propriedade. Para adicionar, clique no botão indicado (1). Para entrar em uma propriedade apenas clique no cartão da propriedade (2). Segure o cartão da propriedade (2) caso queira editá-la ou excluí-la.", R.drawable.terceiratela));
        mList.add(new ScreenItem("Funcionários", "Clique no cartão de funcionários (1) para ir para a tela de funcionários, onde é possível visualizar, adicionar ou excluir. Para adicionar, clique no botão indicado (2). Para excluir segure o cartão do funcionário em questão (3).", R.drawable.quartatela));
        mList.add(new ScreenItem("Culturas", "Essa é a tela de culturas, nela será possível adicionar, visualizar ou excluir uma cultura (caso seja proprietário). Para adicionar, clique no botão indicado (1). Para visualizar apenas clique no cartão da cultura (2). Segure o cartão da cultura (2) caso queira excluí-la.", R.drawable.quintatela));
        mList.add(new ScreenItem("Talhões","Essa é a tela de talhões, você será direcionado para essa tela após selecionar a cultura. O número de talhões é definido automaticamente em função do tamanho da área de cultivo, informada ao criar a cultura. Para adicionar um novo talhão, clique no botão indicado (1). Para selecionar o talhão apenas clique no cartão de talhão (2). Segure o cartão de talhão (2) caso queira editar ou excluir.",R.drawable.sextatela));
        mList.add(new ScreenItem("Pragas atuantes", "Após clicar no cartão da cultura, você será levado para a tela de ações. Ao clicar em pragas atuantes (1), você será levado a uma tela onde é possível adicionar, excluir e identificar a situação em que a praga se encontra. Para adicionar uma praga clique no botão indicado (3), para excluir segure o cartão de praga (4) e para obter informaçõe sobre a situação da praga clique no ícone indicado (2). Você pode clicar no cartão de praga (4) para realizar ações de acordo com a situação da praga.", R.drawable.setimatela));
        mList.add(new ScreenItem("Realizar plano de amostragem", "Após selecionar um talhão, você será levado para a tela de ações. Ao clicar em realizar plano de amostragem (1), você deverá selecionar uma das pragas atuantes em sua cultura através do ícone indicado (2). Para começar o plano de amostragem clique em selecionar praga (3).", R.drawable.oitavatela));
        mList.add(new ScreenItem("Plano de amostragem", "O campo (1) indica informações importantes para realizar o plano de amotragem. Caso tenha dúvida sobre onde encontrar a praga clique no ícone indicado (2). Para realizar o plano de amostragem é necessário amostrar todas as plantas requisitadas neste talhão, a planta atual estará indicada em (3). Para realizar a amostragem selecione encontrado, quando identificar a praga, ou não encontrado (4). Se cometer um erro clique em corrigir (5), para refazer o talhão inteiro segure o botão corrigir (5). Ao término da amostragem clique em finalizar(6), você será direcionado para uma tela com as ações a serem tomadas (7). Clique em (8) para selecionar o método de controle.", R.drawable.nonatela));
        mList.add(new ScreenItem("Aplicar método", "Após realizar o plano de amostragem, caso seja detectada a necessidade de realizar um controle, o aplicativo indicará os métodos de controle para a praga específica. Você poderá aplicar o método de controle após realizar a amostragem (2) ou em outro momento oportuno (1). Caso tenha selecionado aplicar depois (1), na tela de pragas você verá que a praga estará vermelha (3), demonstrando assim a necessidade de controle.", R.drawable.decimatela));
        mList.add(new ScreenItem("Relatórios", "Após ter selecionado o talhão, você será levado para a tela de ações. Ao selecionar gerar relatórios (1), você poderá gerar os tipos de relatório disponíveis. Após selecionar o tipo de relatório, selecione a praga e o relatório será gerado em seguida.", R.drawable.decimaprimeiratela));
        mList.add(new ScreenItem("Tipos de relatório", "Estes são exemplos de relatórios. O relatório pode ser gerado em forma de gráfico ou de lista. No relatório de gráfio clique nos pontos verdes (1) para ver informações da data especificada. O relatório de lista pode ser salvo, clique no botão indicado (2) para salvar o relatório em seu smartphone. Ambos os relatórios servirão como base para o manejo de sua cultura.", R.drawable.decimasegundatela));
        mList.add(new ScreenItem("Função Offline", "Este aplicativo funciona sem a necessidade de internet! Para que você possa utilizar essa função você precisa acessar a tela acima enquanto estiver online. Esta ação deverá ser realizada para cada talhão que você deseja manejar sem internet. Sempre que possível, abra o aplicativo com a conexão de internet ativa, isso garante que seus dados serão salvos no servidor.", R.drawable.decimaterceiratela));


        //setup viewpager
        screenPager = findViewById(R.id.screen_viewpager);
        introViewPagerAdapter = new IntroViewPagerAdapter(this,mList);
        screenPager.setAdapter(introViewPagerAdapter);

        //setup tablayout with viewpager
        tabindicator.setupWithViewPager(screenPager);

        //Botão próximo
        btnNext.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                position = screenPager.getCurrentItem();
                if(position <  mList.size()){

                    position ++;
                    screenPager.setCurrentItem(position);
                }
                if(position == mList.size()-1){ // quando chega na ultima tela

                    carregaUltimaTela();
                }
            }
        });


        //tabLayout add chance listener
        tabindicator.addOnTabSelectedListener(new TabLayout.BaseOnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                if(tab.getPosition()==mList.size()-1){
                    carregaUltimaTela();
                }
            }

            @Override
            public void onTabUnselected(TabLayout.Tab tab) {

            }

            @Override
            public void onTabReselected(TabLayout.Tab tab) {

            }
        });


        //getStarted button click listener
        getStarted.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //open main activity

                Intent main = new Intent(getApplicationContext(), MainActivity.class);
                startActivity(main);

                //also we need to save a boolean value to storage so next time when the user run the app
                //we could know that he is already checked the intro screen activity
                //i'm going to used shared preferences to that process

                savePrefsData();
                finish();

            }
        });

    }

    @Override
    public void onBackPressed() {
        //não faz nada
    }

    private boolean restorePrefData(){
        SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
        Boolean isIntroActivityOpenedBefore = pref.getBoolean("isIntroOpened", false);
        return  isIntroActivityOpenedBefore;
    }

    private void savePrefsData(){
        SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
        SharedPreferences.Editor editor = pref.edit();
        editor.putBoolean("isIntroOpened",true);
        editor.commit();

    }

    //show the getstarted button and hide the indicator and the next button
    public void carregaUltimaTela(){

        btnNext.setVisibility(View.INVISIBLE);
        getStarted.setVisibility(View.VISIBLE);
        tabindicator.setVisibility(View.INVISIBLE);
        //btnAnimation
        getStarted.setAnimation(btnAnim);

    }

}
